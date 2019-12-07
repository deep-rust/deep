use std::collections::HashMap;
use std::ops::AddAssign;

pub struct AccumulateTensors<T> {
    pub table: HashMap<usize, Vec<T>>,
}

impl<T> AccumulateTensors<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T> AccumulateTensors<T>
where
    T: Default + for<'a> AddAssign<&'a T> + 'static,
{
    fn insert(&mut self, slot: usize, tensors: Vec<T>) {
        use std::collections::hash_map::Entry;
        match self.table.entry(slot) {
            Entry::Occupied(mut o) => {
                assert_eq!(
                    o.get().len(),
                    tensors.len(),
                    "AccumulateTensors contains different number of tensors \
                     than what is being added to it for a given op"
                );
                for (at, bt) in o.get_mut().iter_mut().zip(tensors) {
                    *at += &bt;
                }
            }
            Entry::Vacant(v) => {
                v.insert(tensors);
            }
        }
    }
}

impl<T> Extend<(usize, Vec<T>)> for AccumulateTensors<T>
where
    T: Default + for<'a> AddAssign<&'a T> + 'static,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (usize, Vec<T>)>,
    {
        for (slot, tensors) in iter {
            self.insert(slot, tensors);
        }
    }
}

impl<T> Default for AccumulateTensors<T> {
    fn default() -> Self {
        Self {
            table: HashMap::default(),
        }
    }
}
