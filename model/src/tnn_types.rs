use rand::Rng;
use std::ops;

pub const TNNFP16MantWidth: u16 = 7;
pub const TNNFP16ExpWidth: u16 = 8;
pub const TNNFP16MantMask: u16 = (1 << TNNFP16MantWidth) - 1;
pub const TNNFP16ExpMask: u16 = ((1 << TNNFP16ExpWidth) - 1) << TNNFP16MantWidth;
pub const TNNFP16SgnMask: u16 = 1 << (TNNFP16MantWidth + TNNFP16ExpWidth);
pub const TNNFP16Bias: u16 = (1 << (TNNFP16ExpWidth - 1)) - 1;

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct TinyNNFP16(u16);

impl TinyNNFP16 {
    pub const fn new(sgn: bool, exp: u16, mant: u16) -> TinyNNFP16 {
        TinyNNFP16(
            (if sgn { 0x8000 } else { 0 })
                | ((exp << TNNFP16MantWidth) & TNNFP16ExpMask)
                | (mant & TNNFP16MantMask),
        )
    }

    pub fn from_f32(fp: f32) -> TinyNNFP16 {
        if (fp == 0.0) {
            return TinyNNFP16::new(false, 0, 0);
        }

        let sgn = if fp >= 0.0 { false } else { true };

        let unsigned_fp = if sgn { -fp } else { fp };

        let exp = unsigned_fp.log2().floor();
        let mant = (unsigned_fp / exp.exp2()) * (TNNFP16MantWidth as f32).exp2();

        println!("Got {:} {:} {:}", exp, mant, sgn);

        TinyNNFP16::new(
            sgn,
            (exp as i16 + TNNFP16Bias as i16) as u16,
            (mant as u16) & TNNFP16MantMask,
        )
    }
}

pub const TinyNNFP16Zero: TinyNNFP16 = TinyNNFP16::new(false, 0, 0);
pub const TinyNNFP16PosInf: TinyNNFP16 = TinyNNFP16::new(false, (1 << TNNFP16ExpWidth) - 1, 0);
pub const TinyNNFP16NegInf: TinyNNFP16 = TinyNNFP16::new(true, (1 << TNNFP16ExpWidth) - 1, 0);
pub const TinyNNFP16StdNaN: TinyNNFP16 = TinyNNFP16::new(
    true,
    (1 << TNNFP16ExpWidth) - 1,
    (1 << TNNFP16MantWidth) - 1,
);

impl TinyNNFP16 {
    pub fn mant(self) -> u16 {
        self.0 & TNNFP16MantMask
    }

    pub fn mant_with_msb(self) -> u16 {
        self.mant() + (1 << TNNFP16MantWidth)
    }

    pub fn exp(self) -> u16 {
        (self.0 & TNNFP16ExpMask) >> TNNFP16MantWidth
    }

    pub fn sgn(self) -> bool {
        (self.0 & TNNFP16SgnMask) != 0
    }

    pub fn to_f32(self) -> f32 {
        if self.is_zero() {
            return 0.0;
        }

        let mant = (self.mant_with_msb() as f32) / ((1 << TNNFP16MantWidth) as f32);
        let full = mant * (2.0 as f32).powi((self.exp() as i32) - (TNNFP16Bias as i32));

        if self.sgn() {
            -full
        } else {
            full
        }
    }

    pub fn f32_cmp(self, x: f32) -> bool {
        if self.is_nan() {
            return true;
        }

        if self == TinyNNFP16PosInf {
            return x == f32::INFINITY;
        } else if self == TinyNNFP16NegInf {
            return x == f32::NEG_INFINITY;
        }

        if x.is_subnormal() {
            return self.is_zero();
        }

        let self_f32 = self.to_f32();
        let delta = (self_f32 - x).abs();

        if delta == 0.0 {
            return true;
        }

        let epsilon = if self.exp() < TNNFP16MantWidth {
            f32::MIN_POSITIVE
        } else {
            (2.0 as f32)
                .powi((self.exp() as i32) - (TNNFP16Bias as i32) - (TNNFP16MantWidth as i32))
        };

        delta < epsilon
    }

    pub fn is_zero(self) -> bool {
        // Zero is encoded as 0 but make it explicit that all fields are 0
        self.exp() == 0 && self.mant() == 0 && self.sgn() == false
    }

    pub fn as_u16(self) -> u16 {
        self.0
    }

    pub fn is_inf(self) -> bool {
        !self.is_nan() && self.exp() == ((1 << TNNFP16ExpWidth) - 1)
    }

    pub fn is_nan(self) -> bool {
        if self.exp() == 0 || self.exp() == ((1 << TNNFP16ExpWidth) - 1) {
            if self.mant() != 0 {
                return true;
            } else {
                if self.exp() == 0 && self.sgn() == true {
                    return true;
                } else {
                    return false;
                }
            }
        } else {
            return false;
        }
    }
}

impl ops::Mul<TinyNNFP16> for TinyNNFP16 {
    type Output = TinyNNFP16;

    fn mul(self, _rhs: TinyNNFP16) -> TinyNNFP16 {
        if self.is_nan() || _rhs.is_nan() {
            return TinyNNFP16StdNaN;
        }

        if self.is_zero() || _rhs.is_zero() {
            return TinyNNFP16Zero;
        }

        let sgn1 = self.sgn();
        let sgn2 = _rhs.sgn();

        let new_sgn = sgn1 ^ sgn2;

        if self.is_inf() || _rhs.is_inf() {
            if new_sgn {
                return TinyNNFP16NegInf;
            } else {
                return TinyNNFP16PosInf;
            }
        }

        let exp1 = self.exp();
        let exp2 = _rhs.exp();

        let mant1 = self.mant_with_msb() as u32;
        let mant2 = _rhs.mant_with_msb() as u32;

        let new_mant_full = mant1 * mant2;
        let mant_shift = (new_mant_full & (1 << (TNNFP16MantWidth * 2 + 1))) != 0;
        let new_exp = exp1 + exp2 + if mant_shift { 1 } else { 0 };

        if new_exp <= TNNFP16Bias {
            return TinyNNFP16Zero;
        }

        let new_exp = new_exp - TNNFP16Bias;

        if new_exp >= ((1 << TNNFP16ExpWidth) - 1) {
            if new_sgn {
                return TinyNNFP16NegInf;
            } else {
                return TinyNNFP16PosInf;
            }
        }

        let new_mant_shifted = if mant_shift {
            new_mant_full >> TNNFP16MantWidth + 1
        } else {
            new_mant_full >> TNNFP16MantWidth
        };

        TinyNNFP16::new(new_sgn, new_exp, new_mant_shifted as u16)
    }
}

impl ops::Add<TinyNNFP16> for TinyNNFP16 {
    type Output = TinyNNFP16;

    fn add(self, _rhs: TinyNNFP16) -> TinyNNFP16 {
        if self.is_nan() || _rhs.is_nan() {
            return TinyNNFP16StdNaN;
        }

        if self.is_inf() && _rhs.is_inf() {
            if self.sgn() != _rhs.sgn() {
                return TinyNNFP16StdNaN;
            } else {
                return self;
            }
        }

        if self.is_inf() {
            return self;
        }

        if _rhs.is_inf() {
            return _rhs;
        }

        let (mut smaller, larger) = if self.exp() < _rhs.exp() {
            (self, _rhs)
        } else {
            (_rhs, self)
        };

        if smaller.is_zero() {
            return larger;
        }

        if larger.is_zero() {
            return smaller;
        }

        let mut exp_diff = larger.exp() - smaller.exp();

        if exp_diff > TNNFP16MantWidth {
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
            return TinyNNFP16Zero;
        }

        let sum_mant_sgn = sum_mant < 0;
        if sum_mant_sgn {
            sum_mant = -sum_mant;
        }

        let top_bit_index = sum_mant.ilog2() as i32;

        let (final_mant, exp_adjust) = if top_bit_index > TNNFP16MantWidth.into() {
            let exp_adjust: i32 = top_bit_index - (TNNFP16MantWidth as i32);
            (sum_mant >> exp_adjust, exp_adjust)
        } else {
            let exp_adjust: i32 = (TNNFP16MantWidth as i32) - top_bit_index;
            (sum_mant << exp_adjust, -exp_adjust)
        };

        let final_exp = (smaller.exp() as i32) + exp_adjust;

        if final_exp >= ((1 << TNNFP16ExpWidth) - 1) {
            if (sum_mant_sgn) {
                TinyNNFP16NegInf
            } else {
                TinyNNFP16PosInf
            }
        } else if final_exp <= 0 {
            TinyNNFP16Zero
        } else {
            TinyNNFP16::new(sum_mant_sgn, final_exp as u16, final_mant as u16)
        }
    }
}

pub enum RndExpKind {
    FullRND,
    Close,
    Same,
    SmallSmall,
    SmallLarge,
    LargeSmall,
    LargeLarge,
}

pub enum RndMantKind {
    FullRND,
    Zero,
    AllOnes,
}

pub fn gen_rnd_exp_kind() -> RndExpKind {
    match rand::thread_rng().gen_range(0..7) {
        0 => RndExpKind::FullRND,
        1 => RndExpKind::Close,
        2 => RndExpKind::Same,
        3 => RndExpKind::SmallSmall,
        4 => RndExpKind::SmallLarge,
        5 => RndExpKind::LargeSmall,
        6 => RndExpKind::LargeLarge,
        _ => panic!("Unreachable"),
    }
}

pub fn gen_rnd_mant_kind() -> RndMantKind {
    match rand::thread_rng().gen_range(0..3) {
        0 => RndMantKind::FullRND,
        1 => RndMantKind::Zero,
        2 => RndMantKind::AllOnes,
        _ => panic!("Unreachable"),
    }
}

pub fn gen_rnd_mant(rnd_mant_kind: RndMantKind) -> u16 {
    let mut rng = rand::thread_rng();

    match rnd_mant_kind {
        RndMantKind::FullRND => rng.gen_range(0..(1 << TNNFP16MantWidth)),
        RndMantKind::Zero => 0,
        RndMantKind::AllOnes => (1 << TNNFP16MantWidth) - 1,
    }
}

pub fn gen_rnd_fp_operands(rnd_exp_kind: RndExpKind) -> (TinyNNFP16, TinyNNFP16) {
    let mut rng = rand::thread_rng();

    let (exp_a, exp_b): (u16, u16) = match rnd_exp_kind {
        RndExpKind::FullRND => (
            rng.gen_range(0..(1 << TNNFP16ExpWidth)),
            rng.gen_range(0..(1 << TNNFP16ExpWidth)),
        ),
        RndExpKind::Close => {
            let first_exp = rng.gen_range(0..(1 << TNNFP16ExpWidth));
            let other_exp_range_low = if first_exp < 10 { 0 } else { first_exp - 10 };
            let other_exp_range_high = if first_exp > ((1 << TNNFP16ExpWidth) - 10) {
                (1 << TNNFP16ExpWidth)
            } else {
                first_exp + 10
            };
            let second_exp = rng.gen_range(other_exp_range_low..other_exp_range_high);

            (first_exp, second_exp)
        }
        RndExpKind::Same => {
            let exp = rng.gen_range(0..(1 << TNNFP16ExpWidth));
            (exp, exp)
        }
        RndExpKind::SmallSmall => (rng.gen_range(0..30), rng.gen_range(0..30)),
        RndExpKind::SmallLarge => (
            rng.gen_range(0..30),
            rng.gen_range((1 << TNNFP16ExpWidth) - 30..(1 << TNNFP16ExpWidth)),
        ),
        RndExpKind::LargeSmall => (
            rng.gen_range((1 << TNNFP16ExpWidth) - 30..(1 << TNNFP16ExpWidth)),
            rng.gen_range(0..30),
        ),
        RndExpKind::LargeLarge => (
            rng.gen_range((1 << TNNFP16ExpWidth) - 30..(1 << TNNFP16ExpWidth)),
            rng.gen_range((1 << TNNFP16ExpWidth) - 30..(1 << TNNFP16ExpWidth)),
        ),
    };

    return (
        TinyNNFP16::new(rng.gen(), exp_a, gen_rnd_mant(gen_rnd_mant_kind())),
        TinyNNFP16::new(rng.gen(), exp_b, gen_rnd_mant(gen_rnd_mant_kind())),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_mul() {
        let one = TinyNNFP16::new(false, 128, 0);
        let test_num1 = TinyNNFP16::new(false, 128, 0x7f);

        assert_eq!(one.exp(), 128);
        assert_eq!(one.mant(), 0);
        assert_eq!(one.sgn(), false);

        assert_eq!(test_num1 * one, test_num1);
        assert_eq!(test_num1 * test_num1, TinyNNFP16::new(false, 129, 0x7e));
    }
}
