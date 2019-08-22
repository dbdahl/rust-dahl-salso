extern crate num_cpus;

use dahl_partition::*;
use std::slice;

/// A data structure representing a pairwise similarity matrix.
///
pub struct PairwiseSimilarityMatrix {
    data: Vec<f64>,
    n_items: usize,
}

impl PairwiseSimilarityMatrix {
    pub fn new(n_items: usize) -> PairwiseSimilarityMatrix {
        PairwiseSimilarityMatrix {
            data: vec![1.0; n_items * n_items],
            n_items,
        }
    }

    pub fn view(&mut self) -> PairwiseSimilarityMatrixView {
        PairwiseSimilarityMatrixView::from_slice(&mut self.data[..], self.n_items)
    }
}

pub struct PairwiseSimilarityMatrixView<'a> {
    data: &'a mut [f64],
    n_items: usize,
}

impl std::ops::Index<(usize, usize)> for PairwiseSimilarityMatrixView<'_> {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[self.n_items * j + i]
    }
}

impl std::ops::IndexMut<(usize, usize)> for PairwiseSimilarityMatrixView<'_> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[self.n_items * j + i]
    }
}

impl<'a> PairwiseSimilarityMatrixView<'a> {
    pub fn from_slice(data: &'a mut [f64], n_items: usize) -> PairwiseSimilarityMatrixView<'a> {
        assert_eq!(data.len(), n_items * n_items);
        PairwiseSimilarityMatrixView { data, n_items }
    }

    pub unsafe fn from_ptr(data: *mut f64, n_items: usize) -> PairwiseSimilarityMatrixView<'a> {
        let data = slice::from_raw_parts_mut(data, n_items * n_items);
        PairwiseSimilarityMatrixView { data, n_items }
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }

    pub unsafe fn get_unchecked(&self, (i, j): (usize, usize)) -> &f64 {
        self.data.get_unchecked(self.n_items * j + i)
    }

    pub unsafe fn get_unchecked_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        self.data.get_unchecked_mut(self.n_items * j + i)
    }

    pub fn data(&self) -> &[f64] {
        self.data
    }

    pub fn sum_of_triangle(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.n_items {
            for j in 0..i {
                sum += unsafe { self.get_unchecked((i, j)) };
            }
        }
        sum
    }
}

pub fn psm(partitions: &PartitionsHolderView, parallel: bool) -> PairwiseSimilarityMatrix {
    let mut psm = PairwiseSimilarityMatrix::new(partitions.n_items());
    engine(
        partitions.n_partitions(),
        partitions.n_items(),
        parallel,
        partitions,
        &mut psm.view(),
    );
    psm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psm() {
        let mut partitions = PartitionsHolder::new(4);
        partitions.push_partition(&Partition::from("AABB".as_bytes()));
        partitions.push_partition(&Partition::from("AAAB".as_bytes()));
        partitions.push_partition(&Partition::from("ABBB".as_bytes()));
        partitions.push_partition(&Partition::from("AAAB".as_bytes()));
        let partitions_view = partitions.view();
        let psm1 = psm(&partitions_view, true);
        assert_eq!(format!("{:?}", psm1.data), "[1.0, 0.75, 0.5, 0.0, 0.75, 1.0, 0.75, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0, 0.25, 0.5, 1.0]");
        let psm2 = psm(&partitions_view, false);
        assert_eq!(format!("{:?}", psm2.data), "[1.0, 0.75, 0.5, 0.0, 0.75, 1.0, 0.75, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0, 0.25, 0.5, 1.0]");
    }

}

fn engine(
    n_partitions: usize,
    n_items: usize,
    parallel: bool,
    partitions: &PartitionsHolderView,
    psm: &mut PairwiseSimilarityMatrixView,
) -> () {
    if !parallel {
        engine2(n_partitions, n_items, None, partitions, psm);
    } else {
        let n_cores = num_cpus::get();
        let n_pairs = n_items * (n_items - 1) / 2;
        let step_size = n_pairs / n_cores + 1;
        let mut s = 0usize;
        let mut plan = Vec::with_capacity(n_cores + 1);
        plan.push(0);
        for i in 0..n_items {
            if s > step_size {
                plan.push(i);
                s = 0;
            }
            s += i;
        }
        while plan.len() < n_cores + 1 {
            plan.push(n_items);
        }
        crossbeam::scope(|s| {
            for i in 0..n_cores {
                let ptr =
                    unsafe { slice::from_raw_parts_mut(psm.data.as_mut_ptr(), psm.data.len()) };
                let lower = plan[i];
                let upper = plan[i + 1];
                s.spawn(move |_| {
                    let psm2 = &mut PairwiseSimilarityMatrixView::from_slice(ptr, n_items);
                    engine2(n_partitions, n_items, Some(lower..upper), partitions, psm2);
                });
            }
        })
        .unwrap();
    }
}

fn engine2(
    n_partitions: usize,
    n_items: usize,
    range: Option<std::ops::Range<usize>>,
    partitions: &PartitionsHolderView,
    psm: &mut PairwiseSimilarityMatrixView,
) -> () {
    let npf = n_partitions as f64;
    let indices = range.unwrap_or(0..n_items);
    for j in indices {
        for i in 0..j {
            let mut count = 0usize;
            for k in 0..n_partitions {
                unsafe {
                    if partitions.get_unchecked((k, i)) == partitions.get_unchecked((k, j)) {
                        count += 1;
                    }
                }
            }
            let proportion = count as f64 / npf;
            unsafe {
                *psm.get_unchecked_mut((i, j)) = proportion;
                *psm.get_unchecked_mut((j, i)) = proportion;
            }
        }
        unsafe {
            *psm.get_unchecked_mut((j, j)) = 1.0;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_salso__psm(
    n_partitions: i32,
    n_items: i32,
    parallel: i32,
    partitions_ptr: *mut i32,
    psm_ptr: *mut f64,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let partitions = PartitionsHolderView::from_ptr(partitions_ptr, np, ni, true);
    let mut psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    engine(np, ni, parallel != 0, &partitions, &mut psm);
}
