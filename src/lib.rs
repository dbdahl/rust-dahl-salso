#![allow(dead_code)]

#[cfg(test)]
#[macro_use]
extern crate approx;

pub mod loss;
pub mod optimize;
pub mod psm;

use dahl_partition::*;
use std::collections::HashMap;

type LabelType = u16; // usize;
type CountType = u32; // usize;

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

    pub fn plog2p(&self, x: CountType, n: CountType) -> f64 {
        let p = (x as f64) / (n as f64);
        let log2p = unsafe {
            *self.log2n.get_unchecked(x as usize) - *self.log2n.get_unchecked(n as usize)
        };
        p * log2p
    }

    pub fn nlog2n(&self, n: CountType) -> f64 {
        unsafe { *self.nlog2n.get_unchecked(n as usize) }
    }

    pub fn nlog2n_difference(&self, x: CountType) -> f64 {
        unsafe { *self.nlog2n_difference.get_unchecked(x as usize) }
    }
}

pub struct ConfusionMatrix<'a> {
    data: Vec<CountType>,
    labels: &'a [LabelType],
    k1: LabelType,
    k2: LabelType,
}

impl<'a> ConfusionMatrix<'a> {
    pub fn empty(labels: &'a [LabelType], n_clusters: LabelType) -> Self {
        let k1 = n_clusters;
        let k2 = 0;
        let capacity = (1.5 * (k1 as f64 + 1.0) * (k1 as f64 + 1.0)) as usize;
        let data = Vec::with_capacity(capacity);
        let mut x = Self {
            data,
            labels,
            k1,
            k2,
        };
        x.extend();
        x
    }

    pub fn filled(
        fixed_labels: &'a [LabelType],
        fixed_n_clusters: LabelType,
        dynamic_labels: &'a [LabelType],
        dynamic_n_clusters: LabelType,
    ) -> Self {
        let n_items = fixed_labels.len();
        assert_eq!(dynamic_labels.len(), n_items);
        let k1 = fixed_n_clusters;
        let k2 = dynamic_n_clusters;
        let capacity = (k1 as usize + 1) * (k2 as usize + 1);
        let mut x = Self {
            data: vec![0; capacity],
            labels: fixed_labels,
            k1,
            k2,
        };
        x.add_all(dynamic_labels);
        x
    }

    pub fn k1(&self) -> LabelType {
        self.k1
    }

    pub fn k2(&self) -> LabelType {
        self.k2
    }

    pub fn n(&self) -> CountType {
        self.data[0]
    }

    pub fn n1(&self, i: LabelType) -> CountType {
        unsafe { *self.data.get_unchecked(i as usize + 1) }
    }

    pub fn p1(&self, i: LabelType) -> f64 {
        (self.n1(i) as f64) / (self.n() as f64)
    }

    pub fn n2(&self, j: LabelType) -> CountType {
        unsafe {
            *self
                .data
                .get_unchecked((self.k1 as usize + 1) * (j as usize + 1))
        }
    }

    pub fn p2(&self, j: LabelType) -> f64 {
        (self.n2(j) as f64) / (self.n() as f64)
    }

    pub fn n12(&self, i: LabelType, j: LabelType) -> CountType {
        unsafe {
            *self
                .data
                .get_unchecked((self.k1 as usize + 1) * (j as usize + 1) + (i as usize + 1))
        }
    }

    pub fn p12(&self, i: LabelType, j: LabelType) -> f64 {
        (self.n12(i, j) as f64) / (self.n() as f64)
    }

    pub fn new_subset(&mut self) {
        self.k2 += 1;
        self.extend();
    }

    fn extend(&mut self) {
        for _ in 0..=self.k1 {
            self.data.push(0)
        }
    }

    fn add_all(&mut self, clustering: &[LabelType]) {
        for (item_index, label) in clustering.iter().enumerate() {
            self.add_with_index(item_index, *label);
        }
    }

    fn add_with_index(&mut self, item_index: usize, label: LabelType) {
        unsafe {
            *self.data.get_unchecked_mut(0) += 1;
            let offset = (self.k1 as usize + 1) * (label as usize + 1);
            *self.data.get_unchecked_mut(offset) += 1;
            let ii_plus_one = *self.labels.get_unchecked(item_index) as usize + 1;
            *self.data.get_unchecked_mut(ii_plus_one) += 1;
            *self.data.get_unchecked_mut(offset + ii_plus_one) += 1;
        }
    }

    fn remove_with_index(&mut self, item_index: usize, label: LabelType) {
        unsafe {
            *self.data.get_unchecked_mut(0) -= 1;
            let offset = (self.k1 as usize + 1) * (label as usize + 1);
            *self.data.get_unchecked_mut(offset) -= 1;
            let ii_plus_one = *self.labels.get_unchecked(item_index) as usize + 1;
            *self.data.get_unchecked_mut(ii_plus_one) -= 1;
            *self.data.get_unchecked_mut(offset + ii_plus_one) -= 1;
        }
    }

    fn swap_remove(&mut self, killed_label: LabelType, moved_label: LabelType) {
        if killed_label != moved_label {
            for i in 0..=self.k1 {
                unsafe {
                    *self.data.get_unchecked_mut(
                        (self.k1 as usize + 1) * (killed_label as usize + 1) + i as usize,
                    ) = *self.data.get_unchecked(
                        (self.k1 as usize + 1) * (moved_label as usize + 1) + i as usize,
                    )
                }
            }
        }
        self.k2 -= 1;
        self.data
            .truncate((self.k1 as usize + 1) * (self.k2 as usize + 1));
    }
}

pub struct ConfusionMatrices<'a> {
    pub vec: Vec<ConfusionMatrix<'a>>,
}

impl<'a> ConfusionMatrices<'a> {
    pub fn from_draws_empty(draws: &'a Clusterings) -> Self {
        let mut vec = Vec::with_capacity(draws.n_draws);
        for i in 0..draws.n_draws {
            vec.push(ConfusionMatrix::empty(draws.labels(i), draws.n_clusters(i)));
        }
        Self { vec }
    }

    pub fn from_draws_filled(
        draws: &'a Clusterings,
        labels: &'a [LabelType],
        n_clusters: LabelType,
    ) -> Self {
        let mut vec = Vec::with_capacity(draws.n_draws);
        for i in 0..draws.n_draws {
            vec.push(ConfusionMatrix::filled(
                draws.labels(i),
                draws.n_clusters(i),
                labels,
                n_clusters,
            ));
        }
        Self { vec }
    }

    fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        for cm in &mut self.vec {
            cm.new_subset();
        }
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType) {
        partition.add_with_index(i, subset_index as usize);
        for cm in &mut self.vec {
            cm.add_with_index(i, subset_index);
        }
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType {
        let subset_index = partition.label_of(i).unwrap() as LabelType;
        for cm in &mut self.vec {
            cm.remove_with_index(i, subset_index);
        }
        partition.remove_clean_and_relabel(i, |killed_subset_index, moved_subset_index| {
            for cm in &mut self.vec {
                cm.swap_remove(
                    killed_subset_index as LabelType,
                    moved_subset_index as LabelType,
                );
            }
        });
        subset_index
    }
}

pub struct Clusterings {
    n_items: usize,
    n_draws: usize,
    labels: Vec<LabelType>,
    n_clusters: Vec<LabelType>,
}

impl Clusterings {
    pub fn from_i32_column_major_order(original_labels: &[i32], n_items: usize) -> Self {
        let n_draws = original_labels.len() / n_items;
        let mut labels = Vec::with_capacity(n_draws * n_items);
        let mut n_clusters = Vec::with_capacity(n_draws);
        for i in 0..n_draws {
            let mut map = HashMap::new();
            let mut next_new_label = 0;
            for j in 0..n_items {
                let c = *map
                    .entry(unsafe { original_labels.get_unchecked(j * n_draws + i) })
                    .or_insert_with(|| {
                        let c = next_new_label;
                        next_new_label += 1;
                        c
                    });
                labels.push(c);
            }
            n_clusters.push(next_new_label);
        }
        Self {
            n_items,
            n_draws,
            labels,
            n_clusters,
        }
    }

    pub fn labels(&self, index: usize) -> &[LabelType] {
        &self.labels[index * self.n_items..(index + 1) * self.n_items]
    }

    pub fn n_clusters(&self, index: usize) -> LabelType {
        unsafe { *self.n_clusters.get_unchecked(index) }
    }
}
