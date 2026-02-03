//! TNN 16-bit floating point type.
//!
//! This module provides [`TinyNNFP16`], a custom 16-bit floating point format
//! used by the TNN hardware. It is similar to BF16 but not identical.
//!
//! ## Format
//!
//! The 16-bit format is:
//! - 1 sign bit (bit 15)
//! - 8 exponent bits (bits 14-7)
//! - 7 mantissa bits (bits 6-0)
//!
//! ```text
//! Bit: 15 | 14-7 | 6-0
//!      S  | EXP  | MANT
//! ```
//!
//! The exponent uses a bias of 127 (same as IEEE 754 single precision).
//!
//! ## Special Values
//!
//! - Zero: `0x0000`
//! - Positive infinity: sign=0, exp=255, mant=0
//! - Negative infinity: sign=1, exp=255, mant=0
//! - Standard NaN: `0xFFFF` (all ones)

use std::cmp::Ordering;
use std::ops;

/// Mantissa width in bits.
pub const TNNFP16_MANT_WIDTH: u16 = 7;

/// Exponent width in bits.
pub const TNNFP16_EXP_WIDTH: u16 = 8;

/// Bit mask for mantissa (7 bits).
pub const TNNFP16_MANT_MASK: u16 = (1 << TNNFP16_MANT_WIDTH) - 1;

/// Bit mask for exponent (8 bits, shifted).
pub const TNNFP16_EXP_MASK: u16 = ((1 << TNNFP16_EXP_WIDTH) - 1) << TNNFP16_MANT_WIDTH;

/// Bit mask for sign (1 bit at position 15).
pub const TNNFP16_SGN_MASK: u16 = 1 << (TNNFP16_MANT_WIDTH + TNNFP16_EXP_WIDTH);

/// Exponent bias (127).
pub const TNNFP16_BIAS: u16 = (1 << (TNNFP16_EXP_WIDTH - 1)) - 1;

/// TNN 16-bit floating point number.
///
/// A custom floating point format with 1 sign bit, 8 exponent bits, and
/// 7 mantissa bits. Provides arithmetic operations that match TNN hardware
/// behavior.
///
/// # Example
///
/// ```
/// use controller::TinyNNFP16;
///
/// let a = TinyNNFP16::from_f32(2.5);
/// let b = TinyNNFP16::from_f32(1.5);
/// let sum = a + b;
/// assert!((sum.to_f32() - 4.0).abs() < 0.1);
/// ```
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct TinyNNFP16(u16);

impl TinyNNFP16 {
    /// Create a new FP16 value from sign, exponent, and mantissa.
    ///
    /// # Arguments
    ///
    /// * `sgn` - Sign bit (true = negative)
    /// * `exp` - Biased exponent (0-255)
    /// * `mant` - Mantissa (0-127, implicit leading 1)
    pub const fn new(sgn: bool, exp: u16, mant: u16) -> TinyNNFP16 {
        TinyNNFP16(
            (if sgn { 0x8000 } else { 0 })
                | ((exp << TNNFP16_MANT_WIDTH) & TNNFP16_EXP_MASK)
                | (mant & TNNFP16_MANT_MASK),
        )
    }

    /// Convert from f32 to TinyNNFP16.
    ///
    /// Converts the f32 value to the closest representable TinyNNFP16 value.
    /// Values outside the representable range will become infinity or zero.
    pub fn from_f32(fp: f32) -> TinyNNFP16 {
        if fp == 0.0 {
            return TinyNNFP16::zero();
        }

        let sgn = fp < 0.0;
        let unsigned_fp = if sgn { -fp } else { fp };

        let exp = unsigned_fp.log2().floor();
        let mant = (unsigned_fp / exp.exp2()) * (TNNFP16_MANT_WIDTH as f32).exp2();

        TinyNNFP16::new(
            sgn,
            (exp as i16 + TNNFP16_BIAS as i16) as u16,
            (mant as u16) & TNNFP16_MANT_MASK,
        )
    }

    pub const fn zero() -> TinyNNFP16 {
        TinyNNFP16::new(false, 0, 0)
    }

    pub const fn pos_inf() -> TinyNNFP16 {
        TinyNNFP16::new(false, (1 << TNNFP16_EXP_WIDTH) - 1, 0)
    }

    pub const fn neg_inf() -> TinyNNFP16 {
        TinyNNFP16::new(true, (1 << TNNFP16_EXP_WIDTH) - 1, 0)
    }

    pub const fn std_nan() -> TinyNNFP16 {
        TinyNNFP16::new(
            true,
            (1 << TNNFP16_EXP_WIDTH) - 1,
            (1 << TNNFP16_MANT_WIDTH) - 1,
        )
    }

    pub fn mant(self) -> u16 {
        self.0 & TNNFP16_MANT_MASK
    }

    pub fn mant_with_msb(self) -> u16 {
        self.mant() + (1 << TNNFP16_MANT_WIDTH)
    }

    pub fn exp(self) -> u16 {
        (self.0 & TNNFP16_EXP_MASK) >> TNNFP16_MANT_WIDTH
    }

    pub fn sgn(self) -> bool {
        (self.0 & TNNFP16_SGN_MASK) != 0
    }

    pub fn to_f32(self) -> f32 {
        if self.is_zero() {
            return 0.0;
        }

        let mant = (self.mant_with_msb() as f32) / ((1 << TNNFP16_MANT_WIDTH) as f32);
        let full = mant * (2.0_f32).powi((self.exp() as i32) - (TNNFP16_BIAS as i32));

        if self.sgn() {
            -full
        } else {
            full
        }
    }

    pub fn is_zero(self) -> bool {
        self.exp() == 0 && self.mant() == 0 && !self.sgn()
    }

    pub fn as_u16(self) -> u16 {
        self.0
    }

    pub fn from_u16(raw: u16) -> TinyNNFP16 {
        TinyNNFP16(raw)
    }

    pub fn is_inf(self) -> bool {
        !self.is_nan() && self.exp() == ((1 << TNNFP16_EXP_WIDTH) - 1)
    }

    pub fn is_nan(self) -> bool {
        if self.exp() == 0 || self.exp() == ((1 << TNNFP16_EXP_WIDTH) - 1) {
            if self.mant() != 0 {
                return true;
            } else if self.exp() == 0 && self.sgn() {
                return true;
            }
        }
        false
    }
}

impl ops::Mul<TinyNNFP16> for TinyNNFP16 {
    type Output = TinyNNFP16;

    fn mul(self, rhs: TinyNNFP16) -> TinyNNFP16 {
        if self.is_nan() || rhs.is_nan() {
            return TinyNNFP16::std_nan();
        }

        if self.is_zero() || rhs.is_zero() {
            return TinyNNFP16::zero();
        }

        let new_sgn = self.sgn() ^ rhs.sgn();

        if self.is_inf() || rhs.is_inf() {
            return if new_sgn {
                TinyNNFP16::neg_inf()
            } else {
                TinyNNFP16::pos_inf()
            };
        }

        let mant1 = self.mant_with_msb() as u32;
        let mant2 = rhs.mant_with_msb() as u32;

        let new_mant_full = mant1 * mant2;
        let mant_shift = (new_mant_full & (1 << (TNNFP16_MANT_WIDTH * 2 + 1))) != 0;
        let new_exp = self.exp() + rhs.exp() + if mant_shift { 1 } else { 0 };

        if new_exp <= TNNFP16_BIAS {
            return TinyNNFP16::zero();
        }

        let new_exp = new_exp - TNNFP16_BIAS;

        if new_exp >= ((1 << TNNFP16_EXP_WIDTH) - 1) {
            return if new_sgn {
                TinyNNFP16::neg_inf()
            } else {
                TinyNNFP16::pos_inf()
            };
        }

        let new_mant_shifted = if mant_shift {
            new_mant_full >> (TNNFP16_MANT_WIDTH + 1)
        } else {
            new_mant_full >> TNNFP16_MANT_WIDTH
        };

        TinyNNFP16::new(new_sgn, new_exp, new_mant_shifted as u16)
    }
}

impl ops::Add<TinyNNFP16> for TinyNNFP16 {
    type Output = TinyNNFP16;

    fn add(self, rhs: TinyNNFP16) -> TinyNNFP16 {
        if self.is_nan() || rhs.is_nan() {
            return TinyNNFP16::std_nan();
        }

        if self.is_inf() && rhs.is_inf() {
            return if self.sgn() != rhs.sgn() {
                TinyNNFP16::std_nan()
            } else {
                self
            };
        }

        if self.is_inf() {
            return self;
        }

        if rhs.is_inf() {
            return rhs;
        }

        let (smaller, larger) = if self.exp() < rhs.exp() {
            (self, rhs)
        } else {
            (rhs, self)
        };

        if smaller.is_zero() {
            return larger;
        }

        if larger.is_zero() {
            return smaller;
        }

        let exp_diff = larger.exp() - smaller.exp();

        if exp_diff > TNNFP16_MANT_WIDTH {
            return larger;
        }

        let mut larger_mant_shifted = (larger.mant_with_msb() as i32) << exp_diff;

        if larger.sgn() {
            larger_mant_shifted = -larger_mant_shifted;
        }

        let smaller_mant = if smaller.sgn() {
            -(smaller.mant_with_msb() as i32)
        } else {
            smaller.mant_with_msb() as i32
        };

        let mut sum_mant = larger_mant_shifted + smaller_mant;

        if sum_mant == 0 {
            return TinyNNFP16::zero();
        }

        let sum_mant_sgn = sum_mant < 0;
        if sum_mant_sgn {
            sum_mant = -sum_mant;
        }

        let top_bit_index = sum_mant.ilog2() as i32;

        let (final_mant, exp_adjust) = if top_bit_index > TNNFP16_MANT_WIDTH.into() {
            let exp_adjust: i32 = top_bit_index - (TNNFP16_MANT_WIDTH as i32);
            (sum_mant >> exp_adjust, exp_adjust)
        } else {
            let exp_adjust: i32 = (TNNFP16_MANT_WIDTH as i32) - top_bit_index;
            (sum_mant << exp_adjust, -exp_adjust)
        };

        let final_exp = (smaller.exp() as i32) + exp_adjust;

        if final_exp >= ((1 << TNNFP16_EXP_WIDTH) - 1) {
            if sum_mant_sgn {
                TinyNNFP16::neg_inf()
            } else {
                TinyNNFP16::pos_inf()
            }
        } else if final_exp <= 0 {
            TinyNNFP16::zero()
        } else {
            TinyNNFP16::new(sum_mant_sgn, final_exp as u16, final_mant as u16)
        }
    }
}

impl PartialOrd for TinyNNFP16 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            None
        } else if self.0 == other.0 {
            Some(Ordering::Equal)
        } else if self.is_inf() {
            Some(if self.sgn() {
                Ordering::Less
            } else {
                Ordering::Greater
            })
        } else if other.is_inf() {
            Some(if other.sgn() {
                Ordering::Greater
            } else {
                Ordering::Less
            })
        } else if self.is_zero() {
            Some(if other.sgn() {
                Ordering::Greater
            } else {
                Ordering::Less
            })
        } else if other.is_zero() {
            Some(if self.sgn() {
                Ordering::Less
            } else {
                Ordering::Greater
            })
        } else if !self.sgn() && other.sgn() {
            Some(Ordering::Greater)
        } else if self.sgn() && !other.sgn() {
            Some(Ordering::Less)
        } else {
            let pos_result = if self.exp() > other.exp() {
                Ordering::Greater
            } else if self.exp() == other.exp() {
                if self.mant() > other.mant() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            } else {
                Ordering::Less
            };

            Some(if self.sgn() {
                pos_result.reverse()
            } else {
                pos_result
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let zero = TinyNNFP16::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.to_f32(), 0.0);
    }

    #[test]
    fn test_from_f32() {
        let one = TinyNNFP16::from_f32(1.0);
        assert_eq!(one.exp(), 127);
        assert_eq!(one.mant(), 0);
        assert!(!one.sgn());

        let neg_one = TinyNNFP16::from_f32(-1.0);
        assert!(neg_one.sgn());
    }

    #[test]
    fn test_mul() {
        let one = TinyNNFP16::new(false, 127, 0);
        let two = TinyNNFP16::new(false, 128, 0);
        let result = one * two;
        assert_eq!(result.exp(), 128);
        assert_eq!(result.mant(), 0);
    }

    #[test]
    fn test_add() {
        let one = TinyNNFP16::from_f32(1.0);
        let two = TinyNNFP16::from_f32(2.0);
        let result = one + two;
        let expected = TinyNNFP16::from_f32(3.0);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_std_nan() {
        let nan = TinyNNFP16::std_nan();
        assert!(nan.is_nan());
        assert_eq!(nan.as_u16(), 0xFFFF);
    }
}
