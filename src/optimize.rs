extern crate ndarray;
extern crate num_cpus;
extern crate num_traits;
extern crate rand;

use self::ndarray::{Array2, Array3, Axis};
use crate::clustering::Clusterings;
use crate::confusion::{ConfusionMatrices, Log2Cache};
use crate::loss::*;
use crate::*;
use dahl_partition::*;
use dahl_roxido::mk_rng_isaac;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma};
use rand_isaac::IsaacRng;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::f64;
use std::slice;
use std::sync::mpsc;
use std::time::{Duration, SystemTime};

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a.is_nan() {
        Ordering::Greater
    } else if b.is_nan() {
        Ordering::Less
    } else if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn cmp_f64_with_enumeration<T: num_traits::PrimInt>(a: &(T, f64), b: &(T, f64)) -> Ordering {
    if a.1.is_nan() {
        Ordering::Greater
    } else if b.1.is_nan() {
        Ordering::Less
    } else if a.1 < b.1 {
        Ordering::Less
    } else if a.1 > b.1 {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

pub trait Computer {
    fn expected_loss_kernel(&self) -> f64;
    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: LabelType) -> f64;
    fn new_subset(&mut self, partition: &mut Partition);
    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType);
    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType;
}

// Expectation of the Binder loss

#[derive(Debug)]
struct BinderSubsetCalculations {
    committed_loss: f64,
    speculative_loss: f64,
}

pub struct BinderComputer<'a> {
    subsets: Vec<BinderSubsetCalculations>,
    psm: &'a SquareMatrixBorrower<'a>,
}

impl<'a> BinderComputer<'a> {
    pub fn new(psm: &'a SquareMatrixBorrower<'a>) -> Self {
        Self {
            subsets: Vec::new(),
            psm,
        }
    }
    fn expected_loss_from_kernel(psm: &'a SquareMatrixBorrower<'a>, kernel: f64) -> f64 {
        let nif = psm.n_items() as f64;
        (2.0 * kernel + psm.sum_of_triangle()) * 2.0 / (nif * nif)
    }
}

impl<'a> Computer for BinderComputer<'a> {
    fn expected_loss_kernel(&self) -> f64 {
        self.subsets
            .iter()
            .fold(0.0, |s, subset| s + subset.committed_loss)
    }

    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: LabelType) -> f64 {
        self.subsets[subset_index as usize].speculative_loss = partition.subsets()
            [subset_index as usize]
            .items()
            .iter()
            .fold(0.0, |s, j| {
                s + 0.5 - unsafe { *self.psm.get_unchecked((i, *j)) }
            });
        self.subsets[subset_index as usize].speculative_loss
    }

    fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        self.subsets.push(BinderSubsetCalculations {
            committed_loss: 0.0,
            speculative_loss: 0.0,
        });
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType) {
        self.subsets[subset_index as usize].committed_loss +=
            self.subsets[subset_index as usize].speculative_loss;
        partition.add_with_index(i, subset_index as usize);
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType {
        let subset_index = partition.label_of(i).unwrap();
        self.subsets[subset_index].committed_loss -= partition.subsets()[subset_index]
            .items()
            .iter()
            .fold(0.0, |s, j| {
                let jj = *j;
                s + if jj != i {
                    0.5 - unsafe { *self.psm.get_unchecked((i, jj)) }
                } else {
                    0.0
                }
            });
        partition.remove_clean_and_relabel(i, |killed_subset_index, moved_subset_index| {
            self.subsets.swap_remove(killed_subset_index);
            assert_eq!(moved_subset_index, self.subsets.len());
        });
        subset_index as LabelType
    }
}

// Expectation of the Binder loss (alternative implementation)

pub struct Binder2Computer<'a> {
    cms: ConfusionMatrices<'a>,
}

impl<'a> Binder2Computer<'a> {
    pub fn new(draws: &'a Clusterings) -> Self {
        Self {
            cms: ConfusionMatrices::from_draws_empty(draws),
        }
    }
}

impl<'a> Computer for Binder2Computer<'a> {
    fn expected_loss_kernel(&self) -> f64 {
        let mut sum = 0.0;
        let cm = &self.cms.vec[0];
        for j in 0..cm.k2() {
            let n2 = cm.n2(j) as f64;
            sum += n2 * n2;
        }
        sum *= self.cms.vec.len() as f64;
        for cm in &self.cms.vec {
            for i in 0..cm.k1() {
                let n1 = cm.n1(i) as f64;
                sum += n1 * n1;
                for j in 0..cm.k2() {
                    let n12 = cm.n12(i, j) as f64;
                    sum -= 2.0 * n12 * n12;
                }
            }
        }
        let n = self.cms.vec[0].n() as f64;
        sum / (self.cms.vec.len() as f64 * n * n)
    }

    fn speculative_add(
        &mut self,
        _partition: &Partition,
        i: usize,
        subset_index: LabelType,
    ) -> f64 {
        let mut sum = 0.0;
        let n2 = self.cms.vec[0].n2(subset_index) as f64;
        sum += (self.cms.vec.len() as f64) * n2;
        for cm in &self.cms.vec {
            let subset_index_fixed = cm.label(i);
            let n1 = cm.n1(subset_index_fixed) as f64;
            let n12 = cm.n12(subset_index_fixed, subset_index) as f64;
            sum += n1 - 2.0 * n12;
        }
        sum
    }

    fn new_subset(&mut self, partition: &mut Partition) {
        self.cms.new_subset(partition)
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType) {
        self.cms.add_with_index(partition, i, subset_index)
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType {
        self.cms.remove(partition, i)
    }
}

// Expectation of the one minus adjusted Rand index loss

pub struct OneMinusARIComputer<'a> {
    cms: ConfusionMatrices<'a>,
}

impl<'a> OneMinusARIComputer<'a> {
    pub fn new(draws: &'a Clusterings) -> Self {
        Self {
            cms: ConfusionMatrices::from_draws_empty(draws),
        }
    }
}

impl<'a> Computer for OneMinusARIComputer<'a> {
    fn expected_loss_kernel(&self) -> f64 {
        omari_single_kernel(&self.cms)
    }

    fn speculative_add(
        &mut self,
        _partition: &Partition,
        i: usize,
        subset_index: LabelType,
    ) -> f64 {
        for cm in &mut self.cms.vec {
            cm.add_with_index(i, subset_index);
        }
        let result = omari_single_kernel(&self.cms);
        for cm in &mut self.cms.vec {
            cm.remove_with_index(i, subset_index);
        }
        result
    }

    fn new_subset(&mut self, partition: &mut Partition) {
        self.cms.new_subset(partition)
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType) {
        self.cms.add_with_index(partition, i, subset_index)
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType {
        self.cms.remove(partition, i)
    }
}

// First-order approximation of expectation of one minus the adjusted Rand index

#[derive(Debug)]
struct OneMinusARIapproxSubsetCalculations {
    committed_ip: f64,
    committed_i: f64,
    speculative_ip: f64,
    speculative_i: f64,
}

pub struct OneMinusARIapproxComputer<'a> {
    committed_n_items: usize,
    committed_sum_psm: f64,
    speculative_sum_psm: f64,
    subsets: Vec<OneMinusARIapproxSubsetCalculations>,
    psm: &'a SquareMatrixBorrower<'a>,
}

impl<'a> OneMinusARIapproxComputer<'a> {
    pub fn new(psm: &'a SquareMatrixBorrower<'a>) -> Self {
        Self {
            committed_n_items: 0,
            committed_sum_psm: 0.0,
            speculative_sum_psm: f64::NEG_INFINITY,
            subsets: Vec::new(),
            psm,
        }
    }
}

impl<'a> OneMinusARIapproxComputer<'a> {
    fn engine(
        &self,
        speculative_n_items: usize,
        speculative_ip: f64,
        speculative_i: f64,
        speculative_sum_psm: f64,
    ) -> f64 {
        let n_items = speculative_n_items + self.committed_n_items;
        if n_items <= 1 {
            return f64::INFINITY;
        }
        let n_choose_2 = (n_items * (n_items - 1) / 2) as f64;
        let all_ip = speculative_ip + self.subsets.iter().fold(0.0, |s, c| s + c.committed_ip);
        let all_i = speculative_i + self.subsets.iter().fold(0.0, |s, c| s + c.committed_i);
        let all_p = speculative_sum_psm + self.committed_sum_psm;
        let subtractor = all_i * all_p / n_choose_2;
        let numerator = all_ip - subtractor;
        let denominator = 0.5 * (all_i + all_p) - subtractor;
        1.0 - numerator / denominator
    }
}

impl<'a> Computer for OneMinusARIapproxComputer<'a> {
    fn expected_loss_kernel(&self) -> f64 {
        self.engine(0, 0.0, 0.0, 0.0)
    }

    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: LabelType) -> f64 {
        let s = &partition.subsets()[subset_index as usize];
        self.subsets[subset_index as usize].speculative_ip = s
            .items()
            .iter()
            .fold(0.0, |s, j| s + unsafe { *self.psm.get_unchecked((i, *j)) });
        self.subsets[subset_index as usize].speculative_i = s.n_items() as f64;
        if self.speculative_sum_psm == f64::NEG_INFINITY {
            self.speculative_sum_psm = partition.subsets().iter().fold(0.0, |s, subset| {
                // We use the NEG_INFINITY flag to see if we need to do the computation.
                s + subset.items().iter().fold(0.0, |ss, j| {
                    ss + unsafe { *self.psm.get_unchecked((i, *j)) }
                })
            })
        }
        self.engine(
            1,
            self.subsets[subset_index as usize].speculative_ip,
            self.subsets[subset_index as usize].speculative_i,
            self.speculative_sum_psm,
        )
    }

    fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        self.subsets.push(OneMinusARIapproxSubsetCalculations {
            committed_ip: 0.0,
            committed_i: 0.0,
            speculative_ip: 0.0,
            speculative_i: 0.0,
        });
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType) {
        let mut sc = &mut self.subsets[subset_index as usize];
        sc.committed_ip += sc.speculative_ip;
        sc.committed_i += sc.speculative_i;
        self.committed_n_items += 1;
        self.committed_sum_psm += self.speculative_sum_psm;
        self.speculative_sum_psm = f64::NEG_INFINITY;
        partition.add_with_index(i, subset_index as usize);
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType {
        let subset_index = partition.label_of(i).unwrap();
        self.subsets[subset_index].committed_ip -= partition.subsets()[subset_index]
            .items()
            .iter()
            .fold(0.0, |s, j| {
                let jj = *j;
                s + if jj != i {
                    unsafe { *self.psm.get_unchecked((i, jj)) }
                } else {
                    0.0
                }
            });
        self.subsets[subset_index].committed_i -=
            (partition.subsets()[subset_index].n_items() - 1) as f64;
        self.committed_n_items -= 1;
        self.committed_sum_psm -= partition.subsets().iter().fold(0.0, |s, subset| {
            // We use the NEG_INFINITY flag to see if we need to do the computation.
            s + subset.items().iter().fold(0.0, |ss, j| {
                let jj = *j;
                ss + if jj != i {
                    unsafe { *self.psm.get_unchecked((i, jj)) }
                } else {
                    0.0
                }
            })
        });
        partition.remove_clean_and_relabel(i, |killed_subset_index, moved_subset_index| {
            self.subsets.swap_remove(killed_subset_index);
            assert_eq!(moved_subset_index, self.subsets.len());
        });
        subset_index as LabelType
    }
}

// Expectation of the variation of information loss

pub struct VarOfInfoComputer<'a> {
    cms: ConfusionMatrices<'a>,
    cache: &'a Log2Cache,
}

impl<'a> VarOfInfoComputer<'a> {
    pub fn new(draws: &'a Clusterings, cache: &'a Log2Cache) -> Self {
        Self {
            cms: ConfusionMatrices::from_draws_empty(draws),
            cache,
        }
    }
}

impl<'a> Computer for VarOfInfoComputer<'a> {
    fn expected_loss_kernel(&self) -> f64 {
        vi_single_kernel(&self.cms, self.cache)
    }

    fn speculative_add(
        &mut self,
        _partition: &Partition,
        i: usize,
        subset_index: LabelType,
    ) -> f64 {
        let mut sum = 0.0;
        sum += (self.cms.vec.len() as f64)
            * self
                .cache
                .nlog2n_difference(self.cms.vec[0].n2(subset_index));
        for cm in &self.cms.vec {
            let subset_index_fixed = cm.label(i);
            sum -= 2.0
                * self
                    .cache
                    .nlog2n_difference(cm.n12(subset_index_fixed, subset_index));
        }
        sum
    }

    fn new_subset(&mut self, partition: &mut Partition) {
        self.cms.new_subset(partition);
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType) {
        self.cms.add_with_index(partition, i, subset_index)
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType {
        self.cms.remove(partition, i)
    }
}

// Lower bound of the expectation of variation of information loss

#[derive(Debug)]
struct VarOfInfoLBCacheUnit {
    item: usize,
    committed_sum: f64,
    committed_contribution: f64,
    speculative_sum: f64,
    speculative_contribution: f64,
}

#[derive(Debug)]
struct VarOfInfoLBSubsetCalculations {
    cached_units: Vec<VarOfInfoLBCacheUnit>,
    committed_loss: f64,
    speculative_loss: f64,
}

pub struct VarOfInfoLBComputer<'a> {
    subsets: Vec<VarOfInfoLBSubsetCalculations>,
    psm: &'a SquareMatrixBorrower<'a>,
}

impl<'a> VarOfInfoLBComputer<'a> {
    pub fn new(psm: &'a SquareMatrixBorrower<'a>) -> Self {
        Self {
            subsets: Vec::new(),
            psm,
        }
    }
    pub fn expected_loss_from_kernel(psm: &'a SquareMatrixBorrower<'a>, kernel: f64) -> f64 {
        let nif = psm.n_items() as f64;
        (kernel + vilb_expected_loss_constant(psm)) / nif
    }
}

impl<'a> Computer for VarOfInfoLBComputer<'a> {
    fn expected_loss_kernel(&self) -> f64 {
        self.subsets
            .iter()
            .fold(0.0, |s, subset| s + subset.committed_loss)
    }

    fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: LabelType) -> f64 {
        let subset_of_partition = &partition.subsets()[subset_index as usize];
        if subset_of_partition.n_items() == 0 {
            self.subsets[subset_index as usize]
                .cached_units
                .push(VarOfInfoLBCacheUnit {
                    item: i,
                    committed_sum: 0.0,
                    committed_contribution: 0.0,
                    speculative_sum: 1.0,
                    speculative_contribution: 0.0,
                });
            return 0.0;
        }
        for cu in self.subsets[subset_index as usize].cached_units.iter_mut() {
            cu.speculative_sum =
                cu.committed_sum + unsafe { *self.psm.get_unchecked((cu.item, i)) };
            cu.speculative_contribution = cu.speculative_sum.log2();
        }
        let sum = subset_of_partition
            .items()
            .iter()
            .fold(0.0, |s, j| s + unsafe { *self.psm.get_unchecked((i, *j)) })
            + 1.0; // Because self.psm[(i, i)] == 1;
        self.subsets[subset_index as usize]
            .cached_units
            .push(VarOfInfoLBCacheUnit {
                item: i,
                committed_sum: 0.0,
                committed_contribution: 0.0,
                speculative_sum: sum,
                speculative_contribution: sum.log2(),
            });
        let nif = subset_of_partition.n_items() as f64;
        let s1 = (nif + 1.0) * (nif + 1.0).log2();
        let s2 = self.subsets[subset_index as usize]
            .cached_units
            .iter()
            .fold(0.0, |s, cu| s + cu.speculative_contribution);
        self.subsets[subset_index as usize].speculative_loss = s1 - 2.0 * s2;
        self.subsets[subset_index as usize].speculative_loss
            - self.subsets[subset_index as usize].committed_loss
    }

    fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        self.subsets.push(VarOfInfoLBSubsetCalculations {
            cached_units: Vec::new(),
            committed_loss: 0.0,
            speculative_loss: 0.0,
        })
    }

    fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: LabelType) {
        for (index, subset) in self.subsets.iter_mut().enumerate() {
            if index == (subset_index as usize) {
                for cu in subset.cached_units.iter_mut() {
                    cu.committed_sum = cu.speculative_sum;
                    cu.committed_contribution = cu.speculative_contribution;
                }
            } else {
                subset.cached_units.pop();
            }
        }
        self.subsets[subset_index as usize].committed_loss =
            self.subsets[subset_index as usize].speculative_loss;
        partition.add_with_index(i, subset_index as usize);
    }

    fn remove(&mut self, partition: &mut Partition, i: usize) -> LabelType {
        let subset_index = partition.label_of(i).unwrap();
        for cu in self.subsets[subset_index].cached_units.iter_mut() {
            cu.committed_sum -= unsafe { *self.psm.get_unchecked((cu.item, i)) };
            cu.committed_contribution = cu.committed_sum.log2();
        }
        let pos = self.subsets[subset_index]
            .cached_units
            .iter()
            .enumerate()
            .find(|cu| cu.1.item == i)
            .unwrap()
            .0;
        self.subsets[subset_index].cached_units.swap_remove(pos);
        self.subsets[subset_index].committed_loss =
            match partition.subsets()[subset_index].n_items() {
                0 => 0.0,
                ni => {
                    let nif = ni as f64;
                    nif * nif.log2()
                        - 2.0
                            * self.subsets[subset_index]
                                .cached_units
                                .iter()
                                .fold(0.0, |s, cu| s + cu.committed_contribution)
                }
            };
        partition.remove_clean_and_relabel(i, |killed_subset_index, moved_subset_index| {
            self.subsets.swap_remove(killed_subset_index);
            assert_eq!(moved_subset_index, self.subsets.len());
        });
        subset_index as LabelType
    }
}

// Alternative

#[derive(Debug, Clone)]
pub struct WorkingClustering {
    labels: Vec<LabelType>,
    max_clusters: LabelType,
    sizes: Vec<CountType>,
    occupied_clusters: Vec<LabelType>,
    potentially_empty_label: LabelType,
}

impl WorkingClustering {
    pub fn empty(n_items: usize, max_clusters: LabelType) -> Self {
        let max_clusters = max_clusters.max(2);
        let sizes = vec![0; max_clusters as usize];
        let occupied_clusters = Vec::with_capacity(max_clusters as usize);
        Self {
            labels: vec![0; n_items],
            max_clusters,
            sizes,
            occupied_clusters,
            potentially_empty_label: 0,
        }
    }

    pub fn one_cluster(n_items: usize, max_clusters: LabelType) -> Self {
        let max_clusters = max_clusters.max(2);
        let mut sizes = vec![0; max_clusters as usize];
        sizes[0] = n_items as CountType;
        let mut occupied_clusters = Vec::with_capacity(max_clusters as usize);
        occupied_clusters.push(0);
        Self {
            labels: vec![0; n_items],
            max_clusters,
            sizes,
            occupied_clusters,
            potentially_empty_label: 1,
        }
    }

    pub fn random_state<T: Rng>(n_items: usize, rng: &mut T) -> Self {
        Self::from_vector(
            {
                let mut v = Vec::with_capacity(n_items);
                v.resize_with(n_items, || rng.gen_range(0, n_items as LabelType));
                v
            },
            n_items as LabelType,
        )
    }

    pub fn from_slice(labels: &[LabelType], max_clusters: LabelType) -> Self {
        Self::from_vector(labels.to_vec(), max_clusters)
    }

    pub fn from_vector(labels: Vec<LabelType>, max_clusters: LabelType) -> Self {
        let max_clusters = max_clusters.max(2);
        let mut x = Self {
            labels,
            max_clusters,
            sizes: vec![0; max_clusters as usize],
            occupied_clusters: Vec::with_capacity(max_clusters as usize),
            potentially_empty_label: 0,
        };
        for label in &x.labels {
            x.sizes[*label as usize] += 1;
        }
        for (index, size) in x.sizes.iter().enumerate() {
            if *size > 0 {
                x.occupied_clusters.push(index as LabelType)
            }
        }
        x
    }

    pub fn label_of_empty_cluster(&mut self) -> Option<LabelType> {
        if self.occupied_clusters.len() >= self.max_clusters as usize {
            None
        } else {
            if self.sizes[self.potentially_empty_label as usize] == 0 {
                Some(self.potentially_empty_label)
            } else {
                match self.sizes.iter().position(|&size| size == 0) {
                    Some(index) => {
                        self.potentially_empty_label = index as LabelType;
                        Some(self.potentially_empty_label)
                    }
                    None => None,
                }
            }
        }
    }

    pub fn standardize(&self) -> Vec<LabelType> {
        let n_items = self.labels.len();
        let mut labels = Vec::with_capacity(n_items);
        let mut map = HashMap::new();
        let mut next_new_label = 0;
        for j in 0..n_items {
            let c = *map.entry(self.labels[j]).or_insert_with(|| {
                let c = next_new_label;
                next_new_label += 1;
                c
            });
            labels.push(c);
        }
        labels
    }

    pub fn max_clusters(&self) -> LabelType {
        self.max_clusters
    }

    pub fn n_clusters(&self) -> LabelType {
        self.occupied_clusters.len() as LabelType
    }

    pub fn as_slice(&self) -> &[LabelType] {
        &self.labels[..]
    }

    pub fn get(&self, item_index: usize) -> LabelType {
        self.labels[item_index]
    }

    pub unsafe fn get_unchecked(&self, item_index: usize) -> LabelType {
        *self.labels.get_unchecked(item_index)
    }

    pub fn assign(&mut self, item_index: usize, label: LabelType) {
        self.labels[item_index] = label;
        if self.sizes[label as usize] == 0 {
            self.occupied_clusters.push(label);
        }
        self.sizes[label as usize] += 1;
    }

    pub unsafe fn assign_unchecked(&mut self, item_index: usize, label: LabelType) {
        *self.labels.get_unchecked_mut(item_index) = label;
        if *self.sizes.get_unchecked(label as usize) == 0 {
            self.occupied_clusters.push(label);
        }
        *self.sizes.get_unchecked_mut(label as usize) += 1;
    }

    pub fn reassign(&mut self, item_index: usize, new_label: LabelType) {
        let old_label = self.labels[item_index];
        if new_label != old_label {
            self.labels[item_index] = new_label;
            self.sizes[old_label as usize] -= 1;
            if self.sizes[old_label as usize] == 0 {
                self.occupied_clusters.swap_remove(
                    self.occupied_clusters
                        .iter()
                        .position(|x| *x == old_label)
                        .unwrap(),
                );
            }
            if self.sizes[new_label as usize] == 0 {
                self.occupied_clusters.push(new_label);
            }
            self.sizes[new_label as usize] += 1;
        }
    }

    pub unsafe fn reassign_unchecked(&mut self, item_index: usize, new_label: LabelType) {
        let old_label = *self.labels.get_unchecked(item_index);
        if new_label != old_label {
            *self.labels.get_unchecked_mut(item_index) = new_label;
            *self.sizes.get_unchecked_mut(old_label as usize) -= 1;
            if *self.sizes.get_unchecked(old_label as usize) == 0 {
                self.occupied_clusters.swap_remove(
                    self.occupied_clusters
                        .iter()
                        .position(|x| *x == old_label)
                        .unwrap(),
                );
            }
            if *self.sizes.get_unchecked(new_label as usize) == 0 {
                self.occupied_clusters.push(new_label);
            }
            *self.sizes.get_unchecked_mut(new_label as usize) += 1;
        }
    }
}

pub trait LossComputer {
    fn compute_loss(&mut self, state: &WorkingClustering, cms: &Array3<CountType>) -> f64;

    fn change_in_loss(
        &mut self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) -> f64 {
        if from_label_option.is_some() && to_label == from_label_option.unwrap() {
            self.compute_loss(&state, &cms)
        } else {
            let mut state = state.clone();
            let mut cms = cms.clone();
            let n_draws = cms.len_of(Axis(2));
            state.reassign(item_index, to_label);
            for draw_index in 0..n_draws {
                let other_index = draws.label(draw_index, item_index) as usize;
                cms[(to_label as usize + 1, other_index, draw_index)] += 1;
                if from_label_option.is_some() {
                    cms[(
                        from_label_option.unwrap() as usize + 1,
                        other_index,
                        draw_index,
                    )] -= 1;
                }
            }
            self.compute_loss(&state, &cms)
        }
    }

    #[allow(unused_variables)]
    fn decision_callback(
        &mut self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) {
    }
}

// binder

pub(crate) struct BinderLossComputer {}

impl BinderLossComputer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn n_squared(x: CountType) -> f64 {
        let x = x as f64;
        x * x
    }
}

impl LossComputer for BinderLossComputer {
    fn compute_loss(&mut self, state: &WorkingClustering, cms: &Array3<CountType>) -> f64 {
        let mut sum: f64 = state
            .occupied_clusters
            .iter()
            .map(|i| BinderLossComputer::n_squared(state.sizes[*i as usize]))
            .sum();
        let n_draws = cms.len_of(Axis(2));
        sum *= n_draws as f64;
        for draw_index in 0..n_draws {
            for other_index in 0..cms.len_of(Axis(1)) {
                let n = cms[(0, other_index, draw_index)];
                if n > 0 {
                    sum += BinderLossComputer::n_squared(cms[(0, other_index, draw_index)]);
                    for main_label in state.occupied_clusters.iter() {
                        sum -= 2.0
                            * BinderLossComputer::n_squared(
                                cms[(*main_label as usize + 1, other_index, draw_index)],
                            );
                    }
                }
            }
        }
        sum / (n_draws as f64 * BinderLossComputer::n_squared(state.labels.len() as CountType))
    }

    fn change_in_loss(
        &mut self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) -> f64 {
        let offset = if from_label_option.is_some() && to_label == from_label_option.unwrap() {
            1
        } else {
            0
        };
        let n_draws = cms.len_of(Axis(2));
        let to_index = to_label as usize + 1;
        let mut sum = (n_draws as f64) * ((state.sizes[to_index - 1] - offset) as f64) / 2.0;
        for draw_index in 0..n_draws {
            let other_index = draws.label(draw_index, item_index) as usize;
            sum -= (cms[(to_index, other_index, draw_index)] - offset) as f64;
        }
        sum
    }
}

// omARI

pub struct OMARILossComputer {
    n: CountType,
    sum2: f64,
    sums: Array2<f64>,
}

impl OMARILossComputer {
    pub fn new(n_draws: usize) -> Self {
        Self {
            n: 0,
            sum2: 0.0,
            sums: Array2::<f64>::zeros((n_draws, 2)),
        }
    }

    pub fn n_choose_2_times_2(x: CountType) -> f64 {
        let x = x as f64;
        x * (x - 1.0)
    }
}

impl LossComputer for OMARILossComputer {
    fn compute_loss(&mut self, state: &WorkingClustering, cms: &Array3<CountType>) -> f64 {
        // DBD:  This function is completely messed up
        // DBD:  This is a hack since decision_callback isn't working.
        self.n = 0;
        self.sum2 = 0.0;
        self.sums = Array2::<f64>::zeros((cms.len_of(Axis(2)), 2));
        self.n = state.labels.len() as CountType;
        self.sum2 = state
            .occupied_clusters
            .iter()
            .map(|i| OMARILossComputer::n_choose_2_times_2(state.sizes[*i as usize]))
            .sum();
        let n_draws = cms.len_of(Axis(2));
        for draw_index in 0..n_draws {
            for other_index in 0..cms.len_of(Axis(1)) {
                let n = cms[(0, other_index, draw_index)];
                if n > 0 {
                    self.sums[(draw_index, 0)] += OMARILossComputer::n_choose_2_times_2(n);
                    for main_label in state.occupied_clusters.iter() {
                        self.sums[(draw_index, 1)] += OMARILossComputer::n_choose_2_times_2(
                            cms[(*main_label as usize + 1, other_index, draw_index)],
                        );
                    }
                }
            }
        }
        let mut sum = 0.0;
        let sum2 = self.sum2;
        let n_draws = self.sums.len_of(Axis(0));
        for draw_index in 0..n_draws {
            let sum1 = self.sums[(draw_index, 0)];
            let offset = sum1 * sum2 / OMARILossComputer::n_choose_2_times_2(self.n);
            sum += (self.sums[(draw_index, 1)] - offset) / (0.5 * (sum1 + sum2) - offset);
        }
        1.0 - sum / (n_draws as f64)
    }

    fn change_in_loss(
        &mut self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) -> f64 {
        let offset = if from_label_option.is_some() && to_label == from_label_option.unwrap() {
            1
        } else {
            0
        };
        let n_draws = cms.len_of(Axis(2));
        let n2 = (state.sizes[to_label as usize] - offset) as f64;
        let nf = self.n as f64;
        let mut sum = 0.0;
        let to_index = to_label as usize + 1;
        for draw_index in 0..n_draws {
            let mut temp = self.sum2 + n2;
            let other_index = draws.label(draw_index, item_index) as usize;
            let n1 = (cms[(0, other_index, draw_index)] - offset) as f64;
            if n1 > 0.0 {
                let n12 = (cms[(to_index, other_index, draw_index)] - offset) as f64;
                temp += self.sums[(draw_index, 0)] + n1 - 2.0 * (self.sums[(draw_index, 1)] + n12);
            }
            temp /= 0.5 * (self.sum2 + n2 + self.sums[(draw_index, 0)] + n1)
                - (self.sum2 + n2) * (self.sums[(draw_index, 0)] + n1) / ((nf + 1.0) * nf / 2.0);
            sum += temp;
        }
        sum
    }

    fn decision_callback(
        &mut self,
        item_index: usize,
        to_label: LabelType,
        _from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) {
        let n_draws = cms.len_of(Axis(2));
        let n2 = state.sizes[to_label as usize] as f64;
        self.sum2 += n2;
        let to_index = to_label as usize + 1;
        for draw_index in 0..n_draws {
            let other_index = draws.label(draw_index, item_index) as usize;
            let n1 = cms[(0, other_index, draw_index)] as f64;
            if n1 > 0.0 {
                self.sums[(draw_index, 0)] += n1;
                let n12 = cms[(to_index, other_index, draw_index)] as f64;
                self.sums[(draw_index, 1)] += n12;
            }
        }
    }
}

// VI

pub struct VILossComputer<'a> {
    cache: &'a Log2Cache,
}

impl<'a> VILossComputer<'a> {
    pub fn new(cache: &'a Log2Cache) -> Self {
        Self { cache }
    }
}

impl<'a> LossComputer for VILossComputer<'a> {
    fn compute_loss(&mut self, state: &WorkingClustering, cms: &Array3<CountType>) -> f64 {
        let sum2: f64 = state
            .occupied_clusters
            .iter()
            .map(|i| self.cache.nlog2n(state.sizes[*i as usize]))
            .sum();
        let n_draws = cms.len_of(Axis(2));
        let mut sum = 0.0;
        for draw_index in 0..n_draws {
            let mut vi = 0.0;
            for other_index in 0..cms.len_of(Axis(1)) {
                let n = cms[(0, other_index, draw_index)];
                if n > 0 {
                    vi += self.cache.nlog2n(cms[(0, other_index, draw_index)]);
                    for main_label in state.occupied_clusters.iter() {
                        vi -= 2.0
                            * self
                                .cache
                                .nlog2n(cms[(*main_label as usize + 1, other_index, draw_index)]);
                    }
                }
            }
            sum += (vi + sum2) / (state.labels.len() as f64);
        }
        sum / (n_draws as f64)
    }

    fn change_in_loss(
        &mut self,
        item_index: usize,
        to_label: LabelType,
        from_label_option: Option<LabelType>,
        state: &WorkingClustering,
        cms: &Array3<CountType>,
        draws: &Clusterings,
    ) -> f64 {
        let offset = if from_label_option.is_some() && to_label == from_label_option.unwrap() {
            1
        } else {
            0
        };
        let n_draws = cms.len_of(Axis(2));
        let to_index = to_label as usize + 1;
        let mut sum = (n_draws as f64)
            * self
                .cache
                .nlog2n_difference(state.sizes[to_index - 1 as usize] - offset)
            / 2.0;
        for draw_index in 0..n_draws {
            let other_index = draws.label(draw_index, item_index) as usize;
            let n12 = cms[(to_index, other_index, draw_index)] - offset;
            sum -= self.cache.nlog2n_difference(n12);
        }
        sum
    }
}

// Version 2 implementation

pub struct SALSOResults {
    clustering: Vec<usize>,
    expected_loss: f64,
    n_scans: u32,
    prob_exploration: f64,
    n_permutations: u32,
}

impl SALSOResults {
    pub fn new(
        clustering: Vec<usize>,
        expected_loss: f64,
        n_scans: u32,
        prob_exploration: f64,
        n_permutations: u32,
    ) -> Self {
        Self {
            clustering,
            expected_loss,
            n_scans,
            prob_exploration,
            n_permutations,
        }
    }

    pub fn dummy() -> Self {
        SALSOResults::new(vec![0usize; 0], std::f64::INFINITY, 0, 0.0, 0)
    }
}

fn allocation_scan<T: LossComputer>(
    sweetening_scan: bool,
    state: &mut WorkingClustering,
    cms: &mut Array3<CountType>,
    permutation: &Vec<usize>,
    loss_computer: &mut T,
    _p: &SALSOParameters,
    draws: &Clusterings,
) -> bool {
    let mut state_changed = false;
    for item_index in permutation {
        let item_index = *item_index;
        let label_of_empty_cluster = state.label_of_empty_cluster();
        let from_label_option = match sweetening_scan {
            true => Some(state.get(item_index)),
            false => None,
        };
        let iter = state
            .occupied_clusters
            .iter()
            .chain(label_of_empty_cluster.iter())
            .map(|to_label| {
                (
                    *to_label,
                    loss_computer.change_in_loss(
                        item_index,
                        *to_label,
                        from_label_option,
                        &state,
                        &cms,
                        &draws,
                    ),
                )
            });
        let to_label = iter.min_by(cmp_f64_with_enumeration).unwrap().0;
        if !sweetening_scan || to_label != from_label_option.unwrap() {
            loss_computer.decision_callback(
                item_index,
                to_label,
                from_label_option,
                &state,
                &cms,
                &draws,
            );
            if sweetening_scan {
                state.reassign(item_index, to_label);
            } else {
                state.assign(item_index, to_label);
            }
            let from_index = if sweetening_scan {
                from_label_option.unwrap() as usize + 1
            } else {
                0
            };
            let to_index = to_label as usize + 1;
            for draw_index in 0..draws.n_clusterings() {
                let other_index = draws.label(draw_index, item_index) as usize;
                if sweetening_scan {
                    cms[(from_index, other_index, draw_index)] -= 1;
                } else {
                    cms[(0, other_index, draw_index)] += 1;
                }
                cms[(to_index, other_index, draw_index)] += 1;
            }
            state_changed = true;
        }
    }
    state_changed
}

pub fn minimize_once_by_salso_v2<'a, T: LossComputer, U: Rng>(
    loss_computer_factory: Box<dyn Fn() -> T + 'a>,
    draws: &Clusterings,
    p: SALSOParameters,
    stop_time: &SystemTime,
    rng: &mut U,
) -> SALSOResults {
    let n_items = draws.n_items();
    let max_size = match p.max_size {
        0 | 1 => draws.max_clusters(),
        _ => p.max_size,
    };
    let mut permutation: Vec<usize> = (0..p.n_items).collect();
    let mut best = SALSOResults::dummy();
    for permutation_counter in 1..=p.n_permutations {
        let mut state = WorkingClustering::empty(n_items, max_size);
        let mut cms = Array3::<CountType>::zeros((
            state.max_clusters() as usize + 1,
            draws.max_clusters() as usize,
            draws.n_clusterings(),
        ));
        let mut loss_computer = loss_computer_factory();
        // Sequential allocation
        permutation.shuffle(rng);
        allocation_scan(
            false,
            &mut state,
            &mut cms,
            &permutation,
            &mut loss_computer,
            &p,
            draws,
        );
        // Sweetening
        let mut scan_counter = 0;
        let mut state_changed = true;
        while state_changed && scan_counter < p.max_scans {
            scan_counter += 1;
            permutation.shuffle(rng);
            state_changed = allocation_scan(
                true,
                &mut state,
                &mut cms,
                &permutation,
                &mut loss_computer,
                &p,
                draws,
            );
        }
        let expected_loss = loss_computer.compute_loss(&state, &cms);
        if expected_loss < best.expected_loss {
            let clustering = state.standardize().iter().map(|x| *x as usize).collect();
            best = SALSOResults {
                clustering,
                expected_loss,
                n_scans: scan_counter,
                ..best
            }
        }
        if SystemTime::now() > *stop_time {
            best.n_permutations = permutation_counter;
            return best;
        }
    }
    best.n_permutations = p.n_permutations;
    best
}

// General algorithm

fn ensure_empty_subset<U: Computer>(
    partition: &mut Partition,
    computer: &mut U,
    max_label: LabelType,
) {
    match partition.subsets().last() {
        None => computer.new_subset(partition),
        Some(last) => {
            if !last.is_empty() && partition.n_subsets() <= max_label as usize {
                computer.new_subset(partition)
            }
        }
    }
}

fn micro_optimized_allocation<T: Rng, U: Computer>(
    partition: &mut Partition,
    computer: &mut U,
    i: usize,
    probability_of_exploration: f64,
    rng: &mut T,
) -> LabelType {
    let max_label = partition.n_subsets() - 1;
    let mut iter = (0..=max_label)
        .map(|subset_index| computer.speculative_add(partition, i, subset_index as LabelType))
        .enumerate();
    let take_best = if probability_of_exploration > 0.0 {
        rng.gen_range(0.0, 1.0) >= probability_of_exploration
    } else {
        true
    };
    let subset_index = if take_best {
        iter.min_by(cmp_f64_with_enumeration).unwrap().0
    } else {
        let mut first_best = iter.next().unwrap();
        let second_best_option = iter.next();
        if second_best_option.is_none() {
            first_best.0
        } else {
            let mut second_best = second_best_option.unwrap();
            if cmp_f64_with_enumeration(&first_best, &second_best) == Ordering::Greater {
                std::mem::swap(&mut first_best, &mut second_best);
            }
            for tuple in iter {
                if cmp_f64_with_enumeration(&tuple, &second_best) == Ordering::Less {
                    if cmp_f64_with_enumeration(&tuple, &first_best) == Ordering::Less {
                        second_best = first_best;
                        first_best = tuple;
                    } else {
                        second_best = tuple;
                    }
                }
            }
            second_best.0
        }
    };
    computer.add_with_index(partition, i, subset_index as LabelType);
    subset_index as LabelType
}

#[derive(Debug, Copy, Clone)]
pub struct SALSOParameters {
    n_items: usize,
    max_size: LabelType,
    max_scans: u32,
    n_permutations: u32,
    probability_of_exploration_probability_at_zero: f64,
    probability_of_exploration_shape: f64,
    probability_of_exploration_rate: f64,
}

pub fn minimize_once_by_salso<'a, T: Rng, U: Computer>(
    computer_factory: Box<dyn Fn() -> U + 'a>,
    p: SALSOParameters,
    stop_time: &SystemTime,
    rng: &mut T,
) -> SALSOResults {
    let max_label = if p.max_size == 0 {
        LabelType::max_value()
    } else {
        p.max_size - 1
    };
    let probability_of_exploration_distribution = Gamma::new(
        p.probability_of_exploration_shape,
        1.0 / p.probability_of_exploration_rate,
    )
    .unwrap();
    let mut global_minimum = std::f64::INFINITY;
    let mut global_best = Partition::new(p.n_items);
    let mut global_n_scans = 0;
    let mut global_pr_explore = 0.0;
    let mut permutation: Vec<usize> = (0..p.n_items).collect();
    let mut permutations_counter = 0;
    while permutations_counter < p.n_permutations {
        let mut computer = computer_factory();
        let mut partition = Partition::new(p.n_items);
        permutation.shuffle(rng);
        // Initial allocation
        let pr_explore =
            if p.max_scans == 0 || p.probability_of_exploration_probability_at_zero >= 1.0 {
                0.0
            } else {
                if p.probability_of_exploration_probability_at_zero < 0.0 {
                    -p.probability_of_exploration_probability_at_zero
                } else {
                    if rng.gen_range(0.0, 1.0) <= p.probability_of_exploration_probability_at_zero {
                        0.0
                    } else {
                        probability_of_exploration_distribution.sample(rng).min(1.0)
                    }
                }
            };
        for i in 0..p.n_items {
            ensure_empty_subset(&mut partition, &mut computer, max_label);
            let ii = unsafe { *permutation.get_unchecked(i) };
            micro_optimized_allocation(&mut partition, &mut computer, ii, pr_explore, rng);
        }
        // Sweetening scans
        let mut stop_sweetening = false;
        let mut n_scans = 0;
        while n_scans < p.max_scans {
            if n_scans == p.max_scans - 1 {
                stop_sweetening = true;
            }
            permutation.shuffle(rng);
            let mut no_change = true;
            let pr_explore = if stop_sweetening
                || p.probability_of_exploration_probability_at_zero >= 1.0
            {
                0.0
            } else {
                if p.probability_of_exploration_probability_at_zero < 0.0 {
                    -p.probability_of_exploration_probability_at_zero
                } else {
                    if rng.gen_range(0.0, 1.0) <= p.probability_of_exploration_probability_at_zero {
                        0.0
                    } else {
                        probability_of_exploration_distribution.sample(rng).min(1.0)
                    }
                }
            };
            for i in 0..p.n_items {
                ensure_empty_subset(&mut partition, &mut computer, max_label);
                let ii = unsafe { *permutation.get_unchecked(i) };
                let previous_subset_index = computer.remove(&mut partition, ii);
                let subset_index =
                    micro_optimized_allocation(&mut partition, &mut computer, ii, pr_explore, rng);
                if subset_index != previous_subset_index {
                    no_change = false;
                };
            }
            n_scans += 1;
            if no_change {
                if stop_sweetening {
                    break;
                } else {
                    stop_sweetening = true;
                }
            }
        }
        let value = computer.expected_loss_kernel();
        if value < global_minimum {
            global_minimum = value;
            global_best = partition;
            global_n_scans = n_scans;
            global_pr_explore = pr_explore;
        }
        permutations_counter += 1;
        if SystemTime::now() > *stop_time {
            global_best.canonicalize();
            let labels = global_best.labels_via_copying();
            return SALSOResults::new(
                labels,
                global_minimum,
                global_n_scans,
                global_pr_explore,
                permutations_counter,
            );
        }
    }
    // Canonicalize the labels
    global_best.canonicalize();
    let labels = global_best.labels_via_copying();
    SALSOResults::new(
        labels,
        global_minimum,
        global_n_scans,
        global_pr_explore,
        permutations_counter,
    )
}

pub fn minimize_by_salso<T: Rng>(
    pdi: PartitionDistributionInformation,
    loss_function: LossFunction,
    p: SALSOParameters,
    seconds: u64,
    nanoseconds: u32,
    parallel: bool,
    mut rng: &mut T,
) -> SALSOResults {
    let cache = Log2Cache::new(if let LossFunction::VI = loss_function {
        p.n_items
    } else {
        0
    });
    let stop_time = SystemTime::now() + Duration::new(seconds, nanoseconds);
    let result = if !parallel {
        match loss_function {
            LossFunction::Binder => minimize_once_by_salso(
                Box::new(|| BinderComputer::new(pdi.psm())),
                p,
                &stop_time,
                rng,
            ),
            LossFunction::Binder2 => minimize_once_by_salso_v2(
                Box::new(|| BinderLossComputer::new()),
                pdi.draws(),
                p,
                &stop_time,
                rng,
            ),
            LossFunction::OneMinusARI => minimize_once_by_salso_v2(
                Box::new(|| OMARILossComputer::new(pdi.draws().n_clusterings())),
                pdi.draws(),
                p,
                &stop_time,
                rng,
            ),
            LossFunction::OneMinusARIapprox => minimize_once_by_salso(
                Box::new(|| OneMinusARIapproxComputer::new(pdi.psm())),
                p,
                &stop_time,
                rng,
            ),
            LossFunction::VI => minimize_once_by_salso_v2(
                Box::new(|| VILossComputer::new(&cache)),
                pdi.draws(),
                p,
                &stop_time,
                rng,
            ),
            LossFunction::VIlb => minimize_once_by_salso(
                Box::new(|| VarOfInfoLBComputer::new(pdi.psm())),
                p,
                &stop_time,
                rng,
            ),
        }
    } else {
        let (tx, rx) = mpsc::channel();
        let n_cores = num_cpus::get() as u32;
        let p = SALSOParameters {
            n_permutations: (p.n_permutations + n_cores - 1) / n_cores,
            ..p
        };
        let cache_ref = &cache;
        crossbeam::scope(|s| {
            for _ in 0..n_cores {
                let tx = mpsc::Sender::clone(&tx);
                let mut child_rng = IsaacRng::from_rng(&mut rng).unwrap();
                s.spawn(move |_| {
                    let result = match loss_function {
                        LossFunction::Binder => minimize_once_by_salso(
                            Box::new(|| BinderComputer::new(pdi.psm())),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                        LossFunction::Binder2 => minimize_once_by_salso(
                            Box::new(|| Binder2Computer::new(pdi.draws())),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                        LossFunction::OneMinusARI => minimize_once_by_salso(
                            Box::new(|| OneMinusARIComputer::new(pdi.draws())),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                        LossFunction::OneMinusARIapprox => minimize_once_by_salso(
                            Box::new(|| OneMinusARIapproxComputer::new(pdi.psm())),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                        LossFunction::VI => minimize_once_by_salso(
                            Box::new(|| VarOfInfoComputer::new(pdi.draws(), cache_ref)),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                        LossFunction::VIlb => minimize_once_by_salso(
                            Box::new(|| VarOfInfoLBComputer::new(pdi.psm())),
                            p,
                            &stop_time,
                            &mut child_rng,
                        ),
                    };
                    tx.send(result).unwrap();
                });
            }
        })
        .unwrap();
        std::mem::drop(tx); // Because of the cloning in the loop.
        let mut best = SALSOResults::dummy();
        let mut permutations_counter = 0;
        for candidate in rx {
            permutations_counter += candidate.n_permutations;
            if candidate.expected_loss < best.expected_loss {
                best = candidate;
            }
        }
        best.n_permutations = permutations_counter;
        best
    };
    let result = SALSOResults {
        expected_loss: match loss_function {
            LossFunction::Binder => {
                BinderComputer::expected_loss_from_kernel(pdi.psm(), result.expected_loss)
            }
            LossFunction::VIlb => {
                VarOfInfoLBComputer::expected_loss_from_kernel(pdi.psm(), result.expected_loss)
            }
            _ => result.expected_loss,
        },
        ..result
    };
    result
}

pub fn minimize_by_enumeration(
    f: fn(&[LabelType], &SquareMatrixBorrower) -> f64,
    psm: &SquareMatrixBorrower,
) -> Vec<usize> {
    let (tx, rx) = mpsc::channel();
    crossbeam::scope(|s| {
        for iter in Partition::iter_sharded(num_cpus::get() as u32, psm.n_items()) {
            let tx = mpsc::Sender::clone(&tx);
            s.spawn(move |_| {
                let mut working_minimum = std::f64::INFINITY;
                let mut working_minimizer = vec![0usize; psm.n_items()];
                for partition in iter {
                    let part: Vec<LabelType> = partition.iter().map(|x| *x as LabelType).collect();
                    let value = f(&part[..], psm);
                    if value < working_minimum {
                        working_minimum = value;
                        working_minimizer = partition;
                    }
                }
                tx.send(working_minimizer).unwrap();
            });
        }
    })
    .unwrap();
    std::mem::drop(tx); // Because of the cloning in the loop.
    let mut working_minimum = std::f64::INFINITY;
    let mut working_minimizer = vec![0usize; psm.n_items()];
    for partition in rx {
        let part: Vec<LabelType> = partition.iter().map(|x| *x as LabelType).collect();
        let value = f(&part[..], psm);
        if value < working_minimum {
            working_minimum = value;
            working_minimizer = partition;
        }
    }
    working_minimizer
}

#[cfg(test)]
mod tests_optimize {
    use super::rand::thread_rng;
    use super::*;

    #[test]
    fn test_max_scan() {
        let n_items = 5;
        let mut psm = SquareMatrix::identity(n_items);
        let psm_view = &psm.view();
        let p = SALSOParameters {
            n_items,
            max_size: 2,
            max_scans: 10,
            n_permutations: 100,
            probability_of_exploration_probability_at_zero: 0.5,
            probability_of_exploration_shape: 0.5,
            probability_of_exploration_rate: 50.0,
        };
        minimize_by_salso(
            PartitionDistributionInformation::PairwiseSimilarityMatrix(psm_view),
            LossFunction::VIlb,
            p,
            5,
            0,
            false,
            &mut thread_rng(),
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_salso__minimize_by_salso(
    n_items: i32,
    n_draws: i32,
    draws_ptr: *mut i32,
    psm_ptr: *mut f64,
    loss: i32,
    max_size: i32,
    max_scans: i32,
    n_permutations: i32,
    probability_of_exploration_probability_at_zero: f64,
    probability_of_exploration_shape: f64,
    probability_of_exploration_rate: f64,
    seconds: f64,
    parallel: i32,
    results_labels_ptr: *mut i32,
    results_expected_loss_ptr: *mut f64,
    results_scans_ptr: *mut i32,
    results_pr_explore_ptr: *mut f64,
    results_n_permutations_ptr: *mut i32,
    seed_ptr: *const i32, // Assumed length is 32
) {
    let n_items = usize::try_from(n_items).unwrap();
    let nd = usize::try_from(n_draws).unwrap();
    let draws = Clusterings::from_i32_column_major_order(
        PartitionsHolderBorrower::from_ptr(draws_ptr, nd, n_items, true).data(),
        n_items,
    );
    let psm = SquareMatrixBorrower::from_ptr(psm_ptr, n_items);
    let max_size = LabelType::try_from(max_size).unwrap();
    let max_scans = u32::try_from(max_scans).unwrap();
    let n_permutations = u32::try_from(n_permutations).unwrap();
    let (secs, nanos) = if seconds.is_infinite() || seconds < 0.0 {
        (1000 * 365 * 24 * 60 * 60, 0) // 1,000 years
    } else {
        (
            seconds.floor() as u64,
            ((seconds - seconds.floor()) * 1_000_000_000.0).floor() as u32,
        )
    };
    let parallel = parallel != 0;
    let mut rng = mk_rng_isaac(seed_ptr);
    let (loss_function, pdi) = match LossFunction::from_code(loss) {
        Some(loss_function) => match loss_function {
            LossFunction::Binder | LossFunction::OneMinusARIapprox | LossFunction::VIlb => (
                loss_function,
                PartitionDistributionInformation::PairwiseSimilarityMatrix(&psm),
            ),
            LossFunction::Binder2 | LossFunction::OneMinusARI | LossFunction::VI => (
                loss_function,
                PartitionDistributionInformation::Draws(&draws),
            ),
        },
        None => panic!("Unsupported loss method: code = {}", loss),
    };
    let p = SALSOParameters {
        n_items,
        max_size,
        max_scans,
        n_permutations,
        probability_of_exploration_probability_at_zero,
        probability_of_exploration_shape,
        probability_of_exploration_rate,
    };
    let results = minimize_by_salso(pdi, loss_function, p, secs, nanos, parallel, &mut rng);
    let results_slice = slice::from_raw_parts_mut(results_labels_ptr, n_items);
    for (i, v) in results.clustering.iter().enumerate() {
        results_slice[i] = i32::try_from(*v + 1).unwrap();
    }
    *results_expected_loss_ptr = results.expected_loss;
    *results_scans_ptr = i32::try_from(results.n_scans).unwrap();
    *results_pr_explore_ptr = f64::try_from(results.prob_exploration).unwrap();
    *results_n_permutations_ptr = i32::try_from(results.n_permutations).unwrap();
}

#[no_mangle]
pub unsafe extern "C" fn dahl_salso__minimize_by_enumeration(
    n_items: i32,
    psm_ptr: *mut f64,
    loss: i32,
    results_label_ptr: *mut i32,
) {
    let ni = usize::try_from(n_items).unwrap();
    let psm = SquareMatrixBorrower::from_ptr(psm_ptr, ni);
    let f = match LossFunction::from_code(loss) {
        Some(loss_function) => match loss_function {
            LossFunction::Binder => binder_single_kernel,
            LossFunction::Binder2 => panic!("No implementation for binder2."),
            LossFunction::OneMinusARI => panic!("No implementation for omARI."),
            LossFunction::OneMinusARIapprox => omariapprox_single,
            LossFunction::VI => panic!("No implementation for VI."),
            LossFunction::VIlb => vilb_single_kernel,
        },
        None => panic!("Unsupported loss method: code = {}", loss),
    };
    let minimizer = minimize_by_enumeration(f, &psm);
    let results_slice = slice::from_raw_parts_mut(results_label_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v + 1).unwrap();
    }
}
