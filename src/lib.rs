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
    BinderDraws(f64),
    BinderPSM,
    OneMinusARI,
    OneMinusARIapprox,
    VI,
    VIlb,
    NVI,
    ID,
    NID,
}

impl LossFunction {
    fn from_code(x: i32, a: f64) -> Option<LossFunction> {
        match x {
            0 => Some(LossFunction::BinderDraws(a)),
            1 => Some(LossFunction::BinderPSM),
            2 => Some(LossFunction::OneMinusARI),
            3 => Some(LossFunction::OneMinusARIapprox),
            4 => Some(LossFunction::VI),
            5 => Some(LossFunction::VIlb),
            6 => Some(LossFunction::NVI),
            7 => Some(LossFunction::ID),
            8 => Some(LossFunction::NID),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum InitializationMethod {
    SequentialFromEmpty,
    SequentialFromSingletons,
    SampleOne2MaxWithReplacement,
}

impl InitializationMethod {
    fn to_code(&self) -> u32 {
        match self {
            Self::SequentialFromEmpty => 0,
            Self::SequentialFromSingletons => 1,
            Self::SampleOne2MaxWithReplacement => 2,
        }
    }
}
