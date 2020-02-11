#![allow(dead_code)]

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod loss;
pub mod optimize;
pub mod psm;

#[derive(Debug, Copy, Clone)]
pub enum LossFunction {
    Binder,
    AdjRand,
    VIlb,
}

impl LossFunction {
    fn from_code(x: i32) -> Option<LossFunction> {
        match x {
            0 => Some(LossFunction::Binder),
            1 => Some(LossFunction::AdjRand),
            2 => Some(LossFunction::VIlb),
            _ => None,
        }
    }
}
