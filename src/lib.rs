#![allow(dead_code)]

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod loss;
pub mod optimize;
pub mod psm;

use dahl_partition::*;

#[derive(Copy, Clone)]
pub enum PartitionDistributionInformation<'a> {
    Draws(&'a [Partition]),
    PairwiseSimilarityMatrix(&'a SquareMatrixBorrower<'a>),
}

impl<'a> PartitionDistributionInformation<'a> {
    pub fn draws(self) -> &'a [Partition] {
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
    OneMinusARI,
    OneMinusARIapprox,
    VI,
    VIlb,
}

impl LossFunction {
    fn from_code(x: i32) -> Option<LossFunction> {
        match x {
            0 => Some(LossFunction::Binder),
            1 => Some(LossFunction::OneMinusARI),
            2 => Some(LossFunction::OneMinusARIapprox),
            3 => Some(LossFunction::VI),
            4 => Some(LossFunction::VIlb),
            _ => None,
        }
    }
}

pub struct Log2Cache {
    log2n: Vec<f64>,
    nlog2n: Vec<f64>,
    nlog2n_difference: Vec<f64>,
}

impl Log2Cache {
    pub fn new(n: usize) -> Self {
        let mut log2n = Vec::with_capacity(n + 1);
        let mut nlog2n = Vec::with_capacity(n + 1);
        let mut nlog2n_difference = Vec::with_capacity(n + 1);
        log2n.push(0.0);
        nlog2n.push(0.0);
        nlog2n_difference.push(0.0);
        for i in 1..=n {
            let i = i as f64;
            let ilog2i = i.log2();
            log2n.push(ilog2i);
            let ilog2i = i * ilog2i;
            nlog2n.push(ilog2i);
            let i_plus_one = i + 1.0;
            nlog2n_difference.push(i_plus_one * i_plus_one.log2() - ilog2i);
        }
        Self {
            log2n,
            nlog2n,
            nlog2n_difference,
        }
    }

    pub fn plog2p(&self, x: u32, n: u32) -> f64 {
        let p = (x as f64) / (n as f64);
        let log2p = self.log2n[x as usize] - self.log2n[n as usize];
        p * log2p
    }

    pub fn nlog2n(&self, n: u32) -> f64 {
        self.nlog2n[n as usize]
    }

    pub fn nlog2n_difference(&self, x: u32) -> f64 {
        self.nlog2n_difference[x as usize]
    }
}

pub struct ConfusionMatrix<'a> {
    data: Vec<u32>,
    fixed_partition: &'a Partition,
    k1_plus_one: usize,
    k2: usize,
}

impl<'a> ConfusionMatrix<'a> {
    pub fn empty(fixed_partition: &'a Partition) -> Self {
        assert!(fixed_partition.subsets_are_exhaustive());
        let k1_plus_one = fixed_partition.n_subsets() + 1;
        let k2 = 0;
        Self {
            data: vec![0; k1_plus_one * (k2 + 1)],
            fixed_partition,
            k1_plus_one,
            k2,
        }
    }

    pub fn filled(dynamic_partition: &'a Partition, fixed_partition: &'a Partition) -> Self {
        assert!(fixed_partition.subsets_are_exhaustive());
        let n_items = fixed_partition.n_items();
        assert_eq!(dynamic_partition.n_items(), n_items);
        let k1_plus_one = fixed_partition.n_subsets() + 1;
        let k2 = dynamic_partition.n_subsets();
        let mut x = Self {
            data: vec![0; k1_plus_one * (k2 + 1)],
            fixed_partition,
            k1_plus_one,
            k2,
        };
        x.add_all(dynamic_partition);
        x
    }

    pub fn k1(&self) -> usize {
        self.k1_plus_one - 1
    }

    pub fn k2(&self) -> usize {
        self.k2
    }

    pub fn n(&self) -> u32 {
        self.data[0]
    }

    pub fn n1(&self, i: usize) -> u32 {
        self.data[i + 1]
    }

    pub fn p1(&self, i: usize) -> f64 {
        (self.n1(i) as f64) / (self.n() as f64)
    }

    pub fn n2(&self, j: usize) -> u32 {
        self.data[self.k1_plus_one * (j + 1)]
    }

    pub fn p2(&self, j: usize) -> f64 {
        (self.n2(j) as f64) / (self.n() as f64)
    }

    pub fn n12(&self, i: usize, j: usize) -> u32 {
        self.data[self.k1_plus_one * (j + 1) + (i + 1)]
    }

    pub fn p12(&self, i: usize, j: usize) -> f64 {
        (self.n12(i, j) as f64) / (self.n() as f64)
    }

    pub fn new_subset(&mut self) {
        self.k2 += 1;
        self.data.extend(vec![0; self.k1_plus_one].iter());
    }

    fn add_all(&mut self, partition: &Partition) {
        for item_index in 0..partition.n_items() {
            match partition.label_of(item_index) {
                Some(subset_index) => self.add_with_index(item_index, subset_index),
                None => {}
            }
        }
    }

    fn add_with_index(&mut self, item_index: usize, subset_index: usize) {
        self.data[0] += 1;
        let offset = self.k1_plus_one * (subset_index + 1);
        self.data[offset] += 1;
        let ii_plus_one = self.fixed_partition.label_of(item_index).unwrap() + 1;
        self.data[ii_plus_one] += 1;
        self.data[offset + ii_plus_one] += 1;
    }

    fn remove_with_index(&mut self, item_index: usize, subset_index: usize) {
        self.data[0] -= 1;
        let offset = self.k1_plus_one * (subset_index + 1);
        self.data[offset] -= 1;
        let ii_plus_one = self.fixed_partition.label_of(item_index).unwrap() + 1;
        self.data[ii_plus_one] -= 1;
        self.data[offset + ii_plus_one] -= 1;
    }

    fn swap_remove(&mut self, killed_subset_index: usize, moved_subset_index: usize) {
        for i in 0..self.k1_plus_one {
            self.data[self.k1_plus_one * (killed_subset_index + 1) + i] =
                self.data[self.k1_plus_one * (moved_subset_index + 1) + i]
        }
        self.k2 -= 1;
        self.data.truncate(self.k1_plus_one * (self.k2 + 1));
    }
}
