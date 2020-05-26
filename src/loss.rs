use dahl_partition::*;

use crate::*;
use std::slice;

// Expectation of Binder loss

pub fn binder_single_kernel(partition: &[usize], psm: &SquareMatrixBorrower) -> f64 {
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

pub fn binder_single(partition: &[usize], psm: &SquareMatrixBorrower) -> f64 {
    let nif = psm.n_items() as f64;
    binder_single_kernel(partition, psm) * 2.0 / (nif * nif)
}

pub fn binder_multiple(
    partitions: &PartitionsHolderBorrower,
    psm: &SquareMatrixBorrower,
    results: &mut [f64],
) {
    let ni = partitions.n_items();
    assert_eq!(ni, psm.n_items());
    let mut sum_p = 0.0;
    for j in 0..ni {
        for i in 0..j {
            sum_p += unsafe { *psm.get_unchecked((i, j)) };
        }
    }
    let nif = ni as f64;
    let multiplier = 2.0 / (nif * nif);
    for k in 0..partitions.n_partitions() {
        let mut sum = 0.0;
        for j in 0..ni {
            let cj = unsafe { *partitions.get_unchecked((k, j)) };
            for i in 0..j {
                if unsafe { *partitions.get_unchecked((k, i)) == cj } {
                    sum += 1.0 - 2.0 * unsafe { *psm.get_unchecked((i, j)) };
                }
            }
        }
        unsafe { *results.get_unchecked_mut(k) = multiplier * (sum + sum_p) };
    }
}

// Expectation of one minus adjusted Rand index

pub fn omari_single_kernel(cms: &Vec<ConfusionMatrix>) -> f64 {
    pub fn n_choose_2_times_2(x: CountType) -> f64 {
        let x = x as f64;
        x * (x - 1.0)
    }
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let cm = &cms[0];
    for j in 0..cm.k2() {
        sum2 += n_choose_2_times_2(cm.n2(j));
    }
    for cm in cms {
        let mut sum1 = 0.0;
        let mut sum12 = 0.0;
        for i in 0..cm.k1() {
            sum1 += n_choose_2_times_2(cm.n1(i));
            for j in 0..cm.k2() {
                sum12 += n_choose_2_times_2(cm.n12(i, j));
            }
        }
        let offset = sum1 * sum2 / n_choose_2_times_2(cms[0].n());
        sum += (sum12 - offset) / (0.5 * (sum1 + sum2) - offset);
    }
    1.0 - sum / (cms.len() as f64)
}

pub fn omari_single(labels: &[LabelType], n_clusters: LabelType, draws: &Clusterings) -> f64 {
    let cms = ConfusionMatrices::from_draws_filled(draws, labels, n_clusters);
    omari_single_kernel(&cms.vec)
}

pub fn omari_multiple(
    partitions: &PartitionsHolderBorrower,
    draws: &PartitionsHolderBorrower,
    results: &mut [f64],
) {
    let n_items = partitions.n_items();
    assert_eq!(n_items, draws.n_items());
    let clusterings = Clusterings::from_i32_column_major_order(partitions.data(), n_items);
    let draws = Clusterings::from_i32_column_major_order(draws.data(), n_items);
    for k in 0..clusterings.n_draws {
        let omari = omari_single(clusterings.labels(k), clusterings.n_clusters(k), &draws);
        unsafe { *results.get_unchecked_mut(k) = omari };
    }
}

// Approximation of expectation of one minus adjusted Rand index

pub fn omariapprox_single(partition: &[usize], psm: &SquareMatrixBorrower) -> f64 {
    let ni = partition.len();
    assert_eq!(ni, psm.n_items());
    let mut sum_p = 0.0;
    let mut sum_ip = 0.0;
    let mut sum_i = 0.0;
    for j in 0..ni {
        let cj = unsafe { *partition.get_unchecked(j) };
        for i in 0..j {
            let p = unsafe { *psm.get_unchecked((i, j)) };
            sum_p += p;
            if unsafe { *partition.get_unchecked(i) } == cj {
                sum_ip += p;
                sum_i += 1.0;
            }
        }
    }
    let no2 = (ni * (ni - 1) / 2) as f64;
    let correc = (sum_i * sum_p) / no2;
    1.0 - (sum_ip - correc) / (0.5 * (sum_p + sum_i) - correc)
}

pub fn omariapprox_multiple(
    partitions: &PartitionsHolderBorrower,
    psm: &SquareMatrixBorrower,
    results: &mut [f64],
) {
    let ni = partitions.n_items();
    assert_eq!(ni, psm.n_items());
    let no2 = (ni * (ni - 1) / 2) as f64;
    let mut sum_p = 0.0;
    for j in 0..ni {
        for i in 0..j {
            sum_p += unsafe { *psm.get_unchecked((i, j)) };
        }
    }
    for k in 0..partitions.n_partitions() {
        let mut sum_ip = 0.0;
        let mut sum_i = 0.0;
        for j in 0..ni {
            let cj = unsafe { *partitions.get_unchecked((k, j)) };
            for i in 0..j {
                if unsafe { *partitions.get_unchecked((k, i)) } == cj {
                    sum_ip += unsafe { *psm.get_unchecked((i, j)) };
                    sum_i += 1.0;
                }
            }
        }
        let correc = (sum_i * sum_p) / no2;
        let omariapprox = 1.0 - (sum_ip - correc) / (0.5 * (sum_p + sum_i) - correc);
        unsafe { *results.get_unchecked_mut(k) = omariapprox };
    }
}

// Expectation of the variation of information

pub fn vi_single_kernel(cms: &Vec<ConfusionMatrix>, cache: &Log2Cache) -> f64 {
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let cm = &cms[0];
    for j in 0..cm.k2() {
        sum2 += cache.nlog2n(cm.n2(j));
    }
    for cm in cms {
        let mut vi = 0.0;
        for i in 0..cm.k1() {
            vi += cache.nlog2n(cm.n1(i));
            for j in 0..cm.k2() {
                vi -= 2.0 * cache.nlog2n(cm.n12(i, j));
            }
        }
        sum += (vi + sum2) / (cm.n() as f64);
    }
    sum / (cms.len() as f64)
}

pub fn vi_single(
    labels: &[LabelType],
    n_clusters: LabelType,
    draws: &Clusterings,
    cache: &Log2Cache,
) -> f64 {
    let cms = ConfusionMatrices::from_draws_filled(draws, labels, n_clusters);
    vi_single_kernel(&cms.vec, cache)
}

pub fn vi_multiple(
    partitions: &PartitionsHolderBorrower,
    draws: &PartitionsHolderBorrower,
    results: &mut [f64],
) {
    let n_items = partitions.n_items();
    assert_eq!(n_items, draws.n_items());
    let clusterings = Clusterings::from_i32_column_major_order(partitions.data(), n_items);
    let draws = Clusterings::from_i32_column_major_order(draws.data(), n_items);
    let cache = Log2Cache::new(n_items);
    for k in 0..clusterings.n_draws {
        let vi = vi_single(
            clusterings.labels(k),
            clusterings.n_clusters(k),
            &draws,
            &cache,
        );
        unsafe { *results.get_unchecked_mut(k) = vi };
    }
}

// Lower bound of the expectation of the variation of information

pub fn vilb_expected_loss_constant(psm: &SquareMatrixBorrower) -> f64 {
    let ni = psm.n_items();
    let mut s1: f64 = 0.0;
    for i in 0..ni {
        let mut s2: f64 = 0.0;
        for j in 0..ni {
            s2 += unsafe { *psm.get_unchecked((i, j)) };
        }
        s1 += s2.log2();
    }
    s1
}

pub fn vilb_single_kernel(partition: &[usize], psm: &SquareMatrixBorrower) -> f64 {
    let ni = partition.len();
    assert_eq!(ni, psm.n_items());
    let mut sum = 0.0;
    for i in 0..ni {
        let mut s1 = 0u32;
        let mut s2 = 0.0;
        for j in 0..ni {
            if unsafe { *partition.get_unchecked(i) == *partition.get_unchecked(j) } {
                s1 += 1;
                s2 += unsafe { *psm.get_unchecked((i, j)) };
            }
        }
        sum += (s1 as f64).log2() - 2.0 * s2.log2();
    }
    sum
}

pub fn vilb_single_kernel_for_partial_partition(
    partition: &Partition,
    psm: &SquareMatrixBorrower,
) -> f64 {
    let labels = partition.labels();
    let ni = partition.n_items();
    assert_eq!(ni, psm.n_items());
    let mut sum = 0.0;
    for i in 0..ni {
        if labels[i].is_none() {
            continue;
        }
        let mut s1 = 0u32;
        let mut s2 = 0.0;
        for j in 0..ni {
            if labels[j].is_none() {
                continue;
            }
            if partition.label_of(i) == partition.label_of(j) {
                s1 += 1;
                s2 += unsafe { *psm.get_unchecked((i, j)) };
            }
        }
        sum += (s1 as f64).log2() - 2.0 * s2.log2();
    }
    sum
}

pub fn vilb_single(partition: &[usize], psm: &SquareMatrixBorrower) -> f64 {
    (vilb_single_kernel(partition, psm) + vilb_expected_loss_constant(psm)) / (psm.n_items() as f64)
}

pub fn vilb_multiple(
    partitions: &PartitionsHolderBorrower,
    psm: &SquareMatrixBorrower,
    results: &mut [f64],
) {
    let ni = partitions.n_items();
    assert_eq!(ni, psm.n_items());
    let constant = vilb_expected_loss_constant(psm);
    for k in 0..partitions.n_partitions() {
        let mut sum = constant;
        for i in 0..ni {
            let mut s1 = 0u32;
            let mut s3 = 0.0;
            for j in 0..ni {
                if unsafe { *partitions.get_unchecked((k, i)) == *partitions.get_unchecked((k, j)) }
                {
                    s1 += 1;
                    s3 += unsafe { *psm.get_unchecked((i, j)) };
                }
            }
            sum += (s1 as f64).log2() - 2.0 * s3.log2();
        }
        let vilb = sum / (psm.n_items() as f64);
        unsafe { *results.get_unchecked_mut(k) = vilb };
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_salso__expected_loss(
    n_partitions: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    n_draws: i32,
    draws_ptr: *mut i32,
    psm_ptr: *mut f64,
    loss: i32,
    results_ptr: *mut f64,
) {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let nd = n_draws as usize;
    let partitions = PartitionsHolderBorrower::from_ptr(partition_ptr, np, ni, true);
    let draws = PartitionsHolderBorrower::from_ptr(draws_ptr, nd, ni, true);
    let psm = SquareMatrixBorrower::from_ptr(psm_ptr, ni);
    let results = slice::from_raw_parts_mut(results_ptr, np);
    let loss_function = LossFunction::from_code(loss);
    match loss_function {
        Some(LossFunction::Binder) => binder_multiple(&partitions, &psm, results),
        Some(LossFunction::Binder2) => panic!("No implementation for binder2."),
        Some(LossFunction::OneMinusARI) => omari_multiple(&partitions, &draws, results),
        Some(LossFunction::OneMinusARIapprox) => omariapprox_multiple(&partitions, &psm, results),
        Some(LossFunction::VI) => vi_multiple(&partitions, &draws, results),
        Some(LossFunction::VIlb) => vilb_multiple(&partitions, &psm, results),
        None => panic!("Unsupported loss method: {}", loss),
    };
}

#[cfg(test)]
mod tests_loss {
    use super::*;

    #[test]
    fn test_computations() {
        let n_items = 5;
        let mut samples = PartitionsHolder::new(n_items);
        for labels in Partition::iter(n_items) {
            samples.push_partition(&Partition::from(&labels[..]));
        }
        let n_partitions = samples.n_partitions();
        let mut psm = crate::psm::psm(&samples.view(), true);
        let samples_view = &samples.view();
        let psm_view = &psm.view();
        let mut results = vec![0.0; n_partitions];
        binder_multiple(samples_view, psm_view, &mut results[..]);
        for k in 0..n_partitions {
            assert_relative_eq!(
                binder_single(&samples_view.get(k).labels_via_copying()[..], psm_view),
                results[k]
            );
        }
        omari_multiple(samples_view, samples_view, &mut results[..]);
        omariapprox_multiple(samples_view, psm_view, &mut results[..]);
        for k in 0..n_partitions {
            assert_relative_eq!(
                omariapprox_single(&samples_view.get(k).labels_via_copying()[..], psm_view),
                results[k]
            );
        }
        vi_multiple(samples_view, samples_view, &mut results[..]);
        vilb_multiple(samples_view, psm_view, &mut results[..]);
        for k in 0..n_partitions {
            assert_ulps_eq!(
                vilb_single(&samples_view.get(k).labels_via_copying()[..], psm_view),
                results[k]
            );
        }
        for k in 1..n_partitions {
            assert_ulps_eq!(
                ((1.0 / (n_items as f64))
                    * (vilb_single_kernel(&samples_view.get(k).labels_via_copying()[..], psm_view)
                        - vilb_single_kernel(
                            &samples_view.get(k - 1).labels_via_copying()[..],
                            psm_view
                        ))) as f32,
                (results[k] - results[k - 1]) as f32,
            );
        }
    }
}
