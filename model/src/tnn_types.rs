use std::ops;

const TNNFP16MantWidth: u16 = 7;
const TNNFP16ExpWidth: u16 = 8;
const TNNFP16MantMask: u16 = (1 << TNNFP16MantWidth) - 1;
const TNNFP16ExpMask: u16 = ((1 << TNNFP16ExpWidth) - 1) << TNNFP16MantWidth;
const TNNFP16SgnMask: u16 = 1 << (TNNFP16MantWidth + TNNFP16ExpWidth);

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct TinyNNFP16(u16);

impl TinyNNFP16 {
    pub fn new(sgn: bool, exp: u16, mant: u16) -> TinyNNFP16 {
        TinyNNFP16(
            (if sgn { 0x8000 } else { 0 })
                | ((exp << TNNFP16MantWidth) & TNNFP16ExpMask)
                | (mant & TNNFP16MantMask),
        )
    }

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
        let mant = (self.mant_with_msb() as f32) / ((1 << TNNFP16MantWidth) as f32);
        let full = mant * (2.0 as f32).powi((self.exp() as i32) - (1 << TNNFP16ExpWidth - 1));

        if self.sgn() {
            -full
        } else {
            full
        }
    }

    pub fn f32_cmp(self, x: f32) -> bool {
        let self_f32 = self.to_f32();
        let delta = (self_f32 - x).abs();

        if delta == 0.0 {
            return true;
        }

        return delta < (2.0 as f32).powi((self.exp() as i32) - (1 << TNNFP16ExpWidth - 1) - (TNNFP16MantWidth as i32))
    }

    pub fn is_zero(self) -> bool {
        self.exp() == 0
    }

    pub fn as_u16(self) -> u16 {
        self.0
    }
}

//const TinyNNFP16PosZero : TinyNNFP16 = TinyNNFP16::new(false, 0, 0);
//const TinyNNFP16NegZero : TinyNNFP16 = TinyNNFP16::new(true, 0, 0);
//const TinyNNFP16PosInf : TinyNNFP16 = TinyNNFP16::new(false, (TNNFP16ExpWidth << 1) - 1, 0);
//const TinyNNFP16NegInf : TinyNNFP16 = TinyNNFP16::new(true, (TNNFP16ExpWidth << 1) - 1, 0);

impl ops::Mul<TinyNNFP16> for TinyNNFP16 {
    type Output = TinyNNFP16;

    fn mul(self, _rhs: TinyNNFP16) -> TinyNNFP16 {
        let exp1 = self.exp();
        let exp2 = _rhs.exp();

        let mant1 = self.mant_with_msb() as u32;
        let mant2 = _rhs.mant_with_msb() as u32;

        let sgn1 = self.sgn();
        let sgn2 = _rhs.sgn();

        let new_sgn = sgn1 ^ sgn2;

        if exp1 == 0 || exp2 == 0 {
            return TinyNNFP16::new(new_sgn, 0, 0);
        }

        let new_mant_full = mant1 * mant2;
        let mant_shift = (new_mant_full & (1 << (TNNFP16MantWidth * 2 + 1))) != 0;
        let new_exp = exp1 + exp2 - (1 << (TNNFP16ExpWidth - 1)) + if mant_shift { 1 } else { 0 };

        if new_exp >= (1 << TNNFP16ExpWidth) {
            return TinyNNFP16::new(new_sgn, ((1 << TNNFP16ExpWidth) - 1 as u16), 0);
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
            return TinyNNFP16::new(larger.sgn(), 0, 0)
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

        if final_exp >= (1 << TNNFP16ExpWidth) {
            TinyNNFP16::new(sum_mant_sgn, ((1 << TNNFP16ExpWidth) - 1 as u16), 0)
        } else {
            TinyNNFP16::new(sum_mant_sgn, final_exp as u16, final_mant as u16)
        }
    }
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
