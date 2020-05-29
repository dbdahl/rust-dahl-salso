use crate::*;

use std::collections::HashMap;

pub struct Clusterings {
    n_clusterings: usize,
    n_items: usize,
    labels: Vec<LabelType>,
    n_clusters: Vec<LabelType>,
}

impl Clusterings {
    pub fn from_i32_column_major_order(original_labels: &[i32], n_items: usize) -> Self {
        let n_draws = original_labels.len() / n_items;
        let mut labels = Vec::with_capacity(n_draws * n_items);
        let mut n_clusters = Vec::with_capacity(n_draws);
        let mut map = HashMap::new();
        for i in 0..n_draws {
            map.clear();
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
            n_clusterings: n_draws,
            n_items,
            labels,
            n_clusters,
        }
    }

    pub fn label(&self, draw_index: usize, item_index: usize) -> LabelType {
        unsafe {
            *self
                .labels
                .get_unchecked(draw_index * self.n_items + item_index)
        }
    }

    pub fn labels(&self, draw_index: usize) -> &[LabelType] {
        &self.labels[draw_index * self.n_items..(draw_index + 1) * self.n_items]
    }

    pub fn n_clusters(&self, draw_index: usize) -> LabelType {
        unsafe { *self.n_clusters.get_unchecked(draw_index) }
    }

    pub fn n_clusterings(&self) -> usize {
        self.n_clusterings
    }
}
