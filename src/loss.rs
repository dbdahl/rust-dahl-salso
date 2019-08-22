extern crate rand;

use dahl_partition::*;
use crate::psm::PairwiseSimilarityMatrixView;

use std::slice;

pub fn binder_single(partition: &[usize], psm: &PairwiseSimilarityMatrixView) -> f64 {
    let ni = partition.len();
    assert_eq!(ni, psm.n_items());
    let mut sum = 0.0;
    for j in 0..ni {
        for i in 0..j {
            let p = unsafe { *psm.get_unchecked((i, j)) };
            sum += if unsafe { *partition.get_unchecked(i) == *partition.get_unchecked(j) } {
                1.0 - p
            } else {
                p
            }
        }
    }
    sum
}

pub fn binder_multiple(
    partitions: &PartitionsHolderView,
    psm: &PairwiseSimilarityMatrixView,
    results: &mut [f64],
) {
    let ni = partitions.n_items();
    assert_eq!(ni, psm.n_items());
    for k in 0..partitions.n_partitions() {
        let mut sum = 0.0;
        for j in 0..ni {
            for i in 0..j {
                let p = unsafe { *psm.get_unchecked((i, j)) };
                sum += if unsafe {
                    *partitions.get_unchecked((k, i)) == *partitions.get_unchecked((k, j))
                } {
                    1.0 - p
                } else {
                    p
                }
            }
        }
        unsafe { *results.get_unchecked_mut(k) = sum };
    }
}

pub fn vilb_single_kernel(partition: &[usize], psm: &PairwiseSimilarityMatrixView) -> f64 {
    let ni = partition.len();
    assert_eq!(ni, psm.n_items());
    let mut sum = 0.0;
    for i in 0..ni {
        let mut s1 = 0u32;
        let mut s3 = 0.0;
        for j in 0..ni {
            if unsafe { *partition.get_unchecked(i) == *partition.get_unchecked(j) } {
                s1 += 1;
                s3 += unsafe { *psm.get_unchecked((i, j)) };
            }
        }
        sum += f64::from(s1).log2() - 2.0 * s3.log2();
    }
    sum
}

pub fn vilb_multiple(
    partitions: &PartitionsHolderView,
    psm: &PairwiseSimilarityMatrixView,
    results: &mut [f64],
) {
    let ni = partitions.n_items();
    assert_eq!(ni, psm.n_items());
    let sum2 = {
        let mut s1 = 0.0;
        for i in 0..ni {
            let mut s2 = 0.0;
            for j in 0..ni {
                s2 += unsafe { *psm.get_unchecked((i, j)) };
            }
            s1 += s2.log2()
        }
        s1
    };
    for k in 0..partitions.n_partitions() {
        let mut sum = sum2;
        for i in 0..ni {
            let mut s1 = 0u32;
            let mut s2 = 0.0;
            for j in 0..ni {
                if unsafe { *partitions.get_unchecked((k, i)) == *partitions.get_unchecked((k, j)) }
                {
                    s1 += 1;
                    s2 += unsafe { *psm.get_unchecked((i, j)) };
                }
            }
            sum += f64::from(s1).log2() - 2.0 * s2.log2();
        }
        unsafe { *results.get_unchecked_mut(k) = sum / (psm.n_items() as f64) };
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_salso__expected_loss(
    n_partitions: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    psm_ptr: *mut f64,
    loss: i32,
    results_ptr: *mut f64,
) {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let partitions = PartitionsHolderView::from_ptr(partition_ptr, np, ni, true);
    let psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    let results = slice::from_raw_parts_mut(results_ptr, np);
    match loss {
        0 => binder_multiple(&partitions, &psm, results),
        1 => vilb_multiple(&partitions, &psm, results),
        _ => panic!("Unsupported loss method: {}", loss),
    };
}

#[cfg(test)]
mod tests_loss {
    use super::*;
    use crate::distribution::crp::sample;

    #[test]
    fn test_binder() {
        let n_partitions = 1000;
        let n_items = 5;
        let mass = 2.0;
        let mut samples = PartitionsHolder::with_capacity(n_partitions, n_items);
        for _ in 0..n_partitions {
            samples.push_partition(&sample(n_items, mass));
        }
        let mut psm = crate::summary::psm::psm(&samples.view(), true);
        let samples_view = &samples.view();
        let psm_view = &psm.view();
        let mut results = vec![0.0; n_partitions];
        binder_multiple(samples_view, psm_view, &mut results[..]);
        for i in 0..n_items {
            relative_eq!(
                binder_single(&samples_view.get(i).labels_via_copying()[..], psm_view),
                results[i]
            );
        }
        vilb_multiple(samples_view, psm_view, &mut results[..]);
        for i in 1..n_items {
            relative_eq!(
                vilb_single_kernel(&samples_view.get(i).labels_via_copying()[..], psm_view)
                    - vilb_single_kernel(
                        &samples_view.get(i - 1).labels_via_copying()[..],
                        psm_view
                    ),
                results[i] - results[i - 1]
            );
        }
    }
}
