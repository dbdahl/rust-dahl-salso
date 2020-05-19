#![allow(dead_code)]

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod loss;
pub mod optimize;
pub mod psm;

use dahl_partition::*;

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
    cache: Vec<f64>,
}

impl Log2Cache {
    pub fn new(n: usize) -> Self {
        let mut cache = Vec::with_capacity(n + 1);
        cache.push(0.0);
        for i in 1..=n {
            cache.push((i as f64).log2());
        }
        Self { cache }
    }

    pub fn plog2p(&self, x: u32, n: u32) -> f64 {
        let p = (x as f64) / (n as f64);
        let log2p = self.cache[x as usize] - self.cache[n as usize];
        p * log2p
    }

    pub fn n_choose_2_times_2(&self, x: u32) -> f64 {
        (x * (x - 1)) as f64
    }
}

pub struct ConfusionMatrix<'a> {
    data: Vec<u32>,
    fixed_partition: &'a Partition,
    k1_plus_one: usize,
    k2: usize,
    cache: &'a Log2Cache,
}

impl<'a> ConfusionMatrix<'a> {
    pub fn new(
        fixed_partition: &'a Partition,
        dynamic_partition: &'a Partition,
        cache: &'a Log2Cache,
    ) -> Self {
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
            cache,
        };
        for item_index in 0..n_items {
            match dynamic_partition.label_of(item_index) {
                Some(subset_index) => x.tally(item_index, subset_index),
                None => {}
            }
        }
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

    pub fn plogp1(&self, i: usize) -> f64 {
        self.cache.plog2p(self.n1(i), self.n())
    }

    pub fn n2(&self, j: usize) -> u32 {
        self.data[self.k1_plus_one * (j + 1)]
    }

    pub fn p2(&self, j: usize) -> f64 {
        (self.n2(j) as f64) / (self.n() as f64)
    }

    pub fn plogp2(&self, j: usize) -> f64 {
        self.cache.plog2p(self.n2(j), self.n())
    }

    pub fn n12(&self, i: usize, j: usize) -> u32 {
        self.data[self.k1_plus_one * (j + 1) + (i + 1)]
    }

    pub fn p12(&self, i: usize, j: usize) -> f64 {
        (self.n12(i, j) as f64) / (self.n() as f64)
    }

    pub fn plogp12(&self, i: usize, j: usize) -> f64 {
        self.cache.plog2p(self.n12(i, j), self.n())
    }

    pub fn add(&mut self, dynamic_partition: &mut Partition, item_index: usize) {
        let subset_index = dynamic_partition.n_subsets();
        dynamic_partition.add(item_index);
        self.data.extend(vec![0; self.k1_plus_one].iter());
        self.tally(item_index, subset_index);
    }

    pub fn add_with_index(
        &mut self,
        dynamic_partition: &mut Partition,
        item_index: usize,
        subset_index: usize,
    ) {
        dynamic_partition.add_with_index(item_index, subset_index);
        self.tally(item_index, subset_index);
    }

    fn tally(&mut self, item_index: usize, subset_index: usize) {
        self.data[0] += 1;
        let offset = self.k1_plus_one * (subset_index + 1);
        self.data[offset] += 1;
        let ii_plus_one = self.fixed_partition.label_of(item_index).unwrap() + 1;
        self.data[ii_plus_one] += 1;
        self.data[offset + ii_plus_one] += 1;
    }
}
