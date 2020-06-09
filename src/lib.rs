#![allow(dead_code)]

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod clustering;
pub mod log2cache;
pub mod loss;
pub mod optimize;
pub mod psm;

use crate::clustering::Clusterings;
use dahl_partition::*;

type LabelType = u16; // u8; // usize;
type CountType = u32; // u16; // usize;

#[derive(Copy, Clone)]
pub enum PartitionDistributionInformation<'a> {
    Draws(&'a Clusterings),
    PairwiseSimilarityMatrix(&'a SquareMatrixBorrower<'a>),
}

impl<'a> PartitionDistributionInformation<'a> {
    pub fn draws(self) -> &'a Clusterings {
        match self {
            PartitionDistributionInformation::Draws(d) => d,
            _ => panic!("Not available."),
        }
    }
    pub fn psm(self) -> &'a SquareMatrixBorrower<'a> {
        match self {
            PartitionDistributionInformation::PairwiseSimilarityMatrix(p) => p,
            _ => panic!("Not available."),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum LossFunction {
    Binder,
    Binder2,
    OneMinusARI,
    OneMinusARIapprox,
    VI,
    VIlb,
}

impl LossFunction {
    fn from_code(x: i32) -> Option<LossFunction> {
        match x {
            0 => Some(LossFunction::Binder),
            1 => Some(LossFunction::Binder2),
            2 => Some(LossFunction::OneMinusARI),
            3 => Some(LossFunction::OneMinusARIapprox),
            4 => Some(LossFunction::VI),
            5 => Some(LossFunction::VIlb),
            _ => None,
        }
    }
}
