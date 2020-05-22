#![allow(dead_code)]

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod loss;
pub mod optimize;
pub mod psm;

use dahl_partition::*;
use std::collections::HashMap;

#[derive(Copy, Clone)]
pub enum PartitionDistributionInformation<'a> {
    Draws(&'a Vec<ClusterLabels>),
    PairwiseSimilarityMatrix(&'a SquareMatrixBorrower<'a>),
}

impl<'a> PartitionDistributionInformation<'a> {
    pub fn draws(self) -> &'a Vec<ClusterLabels> {
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
        for i in 1..=n {
            let i = i as f64;
            let log2i = i.log2();
            log2n.push(log2i);
            let ilog2i = i * log2i;
            let ilog2i_last = *nlog2n.last().unwrap();
            nlog2n.push(ilog2i);
            nlog2n_difference.push(ilog2i - ilog2i_last);
        }
        Self {
            log2n,
            nlog2n,
            nlog2n_difference,
        }
    }

    pub fn plog2p(&self, x: usize, n: usize) -> f64 {
        let p = (x as f64) / (n as f64);
        let log2p = self.log2n[x] - self.log2n[n];
        p * log2p
    }

    pub fn nlog2n(&self, n: usize) -> f64 {
        self.nlog2n[n]
    }

    pub fn nlog2n_difference(&self, x: usize) -> f64 {
        self.nlog2n_difference[x]
    }
}

pub struct ConfusionMatrix<'a> {
    data: Vec<usize>,
    fixed_partition: &'a ClusterLabels,
    k1_plus_one: usize,
    k2: usize,
}

impl<'a> ConfusionMatrix<'a> {
    pub fn empty(fixed_partition: &'a ClusterLabels) -> Self {
        let k1_plus_one = fixed_partition.n_clusters + 1;
        let k2 = 0;
        let data = Vec::with_capacity(2 * k1_plus_one * k1_plus_one);
        let mut x = Self {
            data,
            fixed_partition,
            k1_plus_one,
            k2,
        };
        x.extend();
        x
    }

    pub fn filled(
        dynamic_partition: &'a ClusterLabels,
        fixed_partition: &'a ClusterLabels,
    ) -> Self {
        let n_items = fixed_partition.labels.len();
        assert_eq!(dynamic_partition.labels.len(), n_items);
        let k1_plus_one = fixed_partition.n_clusters + 1;
        let k2 = dynamic_partition.n_clusters;
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

    pub fn n(&self) -> usize {
        self.data[0]
    }

    pub fn n1(&self, i: usize) -> usize {
        self.data[i + 1]
    }

    pub fn p1(&self, i: usize) -> f64 {
        (self.n1(i) as f64) / (self.n() as f64)
    }

    pub fn n2(&self, j: usize) -> usize {
        self.data[self.k1_plus_one * (j + 1)]
    }

    pub fn p2(&self, j: usize) -> f64 {
        (self.n2(j) as f64) / (self.n() as f64)
    }

    pub fn n12(&self, i: usize, j: usize) -> usize {
        self.data[self.k1_plus_one * (j + 1) + (i + 1)]
    }

    pub fn p12(&self, i: usize, j: usize) -> f64 {
        (self.n12(i, j) as f64) / (self.n() as f64)
    }

    pub fn new_subset(&mut self) {
        self.k2 += 1;
        self.extend();
    }

    fn extend(&mut self) {
        for _ in 0..self.k1_plus_one {
            self.data.push(0)
        }
    }

    fn add_all(&mut self, partition: &ClusterLabels) {
        for (item_index, label) in partition.labels.iter().enumerate() {
            self.add_with_index(item_index, *label);
        }
    }

    fn add_with_index(&mut self, item_index: usize, label: usize) {
        self.data[0] += 1;
        let offset = self.k1_plus_one * (label + 1);
        self.data[offset] += 1;
        let ii_plus_one = self.fixed_partition.labels[item_index] + 1;
        self.data[ii_plus_one] += 1;
        self.data[offset + ii_plus_one] += 1;
    }

    fn remove_with_index(&mut self, item_index: usize, label: usize) {
        self.data[0] -= 1;
        let offset = self.k1_plus_one * (label + 1);
        self.data[offset] -= 1;
        let ii_plus_one = self.fixed_partition.labels[item_index] + 1;
        self.data[ii_plus_one] -= 1;
        self.data[offset + ii_plus_one] -= 1;
    }

    fn swap_remove(&mut self, killed_label: usize, moved_label: usize) {
        if killed_label != moved_label {
            for i in 0..self.k1_plus_one {
                self.data[self.k1_plus_one * (killed_label + 1) + i] =
                    self.data[self.k1_plus_one * (moved_label + 1) + i]
            }
        }
        self.k2 -= 1;
        self.data.truncate(self.k1_plus_one * (self.k2 + 1));
    }
}

pub struct ClusterLabels {
    labels: Vec<usize>,
    n_clusters: usize,
}

pub fn standardize_labels(labels: &[i32], n_items: usize) -> Vec<ClusterLabels> {
    let n_samples = labels.len() / n_items;
    let mut new_labels_collection = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut new_labels = Vec::with_capacity(n_items);
        let mut map = HashMap::new();
        let mut next_new_label = 0;
        for j in 0..n_items {
            let c = *map.entry(labels[j * n_samples + i]).or_insert_with(|| {
                let c = next_new_label;
                next_new_label += 1;
                c
            });
            new_labels.push(c);
        }
        new_labels_collection.push(ClusterLabels {
            labels: new_labels,
            n_clusters: next_new_label,
        });
    }
    new_labels_collection
}
