use crate::{
    arena::Arena,
    keys::{Key, Spur},
    util::{Iter, Strings},
    Rodeo, RodeoReader,
};
use core::{marker::PhantomData, ops::Index};

compile! {
    if #[feature = "no-std"] {
        use alloc::vec::Vec;
    }
}

/// A read-only view of a [`Rodeo`] or [`ThreadedRodeo`] that allows contention-free access to interned strings
/// with only key to string resolution
///
/// The key type is the same as the `Rodeo` or `ThreadedRodeo` that created it
///
/// [`Rodeo`]: crate::Rodeo
/// [`ThreadedRodeo`]: crate::ThreadedRodeo
#[derive(Debug)]
pub struct RodeoResolver<K = Spur> {
    /// Vector of strings mapped to key indexes that allows key to string resolution
    pub(crate) strings: Vec<&'static [u16]>,
    /// The arena that contains all the strings
    ///
    /// This is not touched, but *must* be kept since every string in `self.strings`
    /// points to it
    __arena: Arena,
    /// The type of the key
    __key: PhantomData<K>,
}

impl<K> RodeoResolver<K> {
    /// Creates a new RodeoResolver
    ///
    /// # Safety
    ///
    /// The references inside of `strings` must be absolutely unique, meaning
    /// that no other references to those strings exist
    ///
    pub(crate) unsafe fn new(strings: Vec<&'static [u16]>, arena: Arena) -> Self {
        Self {
            strings,
            __arena: arena,
            __key: PhantomData,
        }
    }

    /// Resolves a string by its key. Only keys made by the current Resolver or the creator
    /// of the current Resolver may be used
    ///
    /// # Panics
    ///
    /// Panics if the key is out of bounds
    ///
    /// # Example
    ///
    /// ```rust
    /// use lasso_u16::Rodeo;
    ///
    /// // ThreadedRodeo is interchangeable for Rodeo here
    /// let mut rodeo = Rodeo::default();
    /// let key = rodeo.get_or_intern("Strings of things with wings and dings");
    ///
    /// let rodeo = rodeo.into_resolver();
    /// assert_eq!("Strings of things with wings and dings", rodeo.resolve(&key));
    /// ```
    ///
    /// [`Key`]: crate::Key
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn resolve<'a>(&'a self, key: &K) -> &'a [u16]
    where
        K: Key,
    {
        // Safety: The call to get_unchecked's safety relies on the Key::into_usize impl
        // being symmetric and the caller having not fabricated a key. If the impl is sound
        // and symmetric, then it will succeed, as the usize used to create it is a valid
        // index into self.strings
        unsafe {
            assert!(key.into_usize() < self.strings.len());

            self.strings.get_unchecked(key.into_usize())
        }
    }

    /// Resolves a string by its key, returning `None` if the key is out of bounds. Only keys
    /// made by the current Resolver or the creator of the current Resolver may be used
    ///
    /// # Example
    ///
    /// ```rust
    /// use lasso_u16::Rodeo;
    ///
    /// // ThreadedRodeo is interchangeable for Rodeo here
    /// let mut rodeo = Rodeo::default();
    /// let key = rodeo.get_or_intern("Strings of things with wings and dings");
    ///
    /// let rodeo = rodeo.into_resolver();
    /// assert_eq!(Some("Strings of things with wings and dings"), rodeo.try_resolve(&key));
    /// ```
    ///
    /// [`Key`]: crate::Key
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn try_resolve<'a>(&'a self, key: &K) -> Option<&'a [u16]>
    where
        K: Key,
    {
        // Safety: The call to get_unchecked's safety relies on the Key::into_usize impl
        // being symmetric and the caller having not fabricated a key. If the impl is sound
        // and symmetric, then it will succeed, as the usize used to create it is a valid
        // index into self.strings
        unsafe {
            if key.into_usize() < self.strings.len() {
                Some(self.strings.get_unchecked(key.into_usize()))
            } else {
                None
            }
        }
    }

    /// Resolves a string by its key without preforming bounds checks
    ///
    /// # Safety
    ///
    /// The key must be valid for the current interner
    ///
    /// # Example
    ///
    /// ```rust
    /// use lasso_u16::Rodeo;
    ///
    /// // ThreadedRodeo is interchangeable for Rodeo here
    /// let mut rodeo = Rodeo::default();
    /// let key = rodeo.get_or_intern("Strings of things with wings and dings");
    ///
    /// let rodeo = rodeo.into_resolver();
    /// unsafe {
    ///     assert_eq!("Strings of things with wings and dings", rodeo.resolve_unchecked(&key));
    /// }
    /// ```
    ///
    /// [`Key`]: crate::Key
    #[cfg_attr(feature = "inline-more", inline)]
    pub unsafe fn resolve_unchecked<'a>(&'a self, key: &K) -> &'a [u16]
    where
        K: Key,
    {
        self.strings.get_unchecked(key.into_usize())
    }

    /// Returns `true` if the given key exists in the current interner
    ///
    /// # Example
    ///
    /// ```rust
    /// use lasso_u16::Rodeo;
    /// # use lasso_u16::{Key, Spur};
    ///
    /// let mut rodeo = Rodeo::default();
    /// let key = rodeo.get_or_intern("Strings of things with wings and dings");
    /// # let key_that_doesnt_exist = Spur::try_from_usize(1000).unwrap();
    ///
    /// let rodeo = rodeo.into_resolver();
    /// assert!(rodeo.contains_key(&key));
    /// assert!(!rodeo.contains_key(&key_that_doesnt_exist));
    /// ```
    ///
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn contains_key(&self, key: &K) -> bool
    where
        K: Key,
    {
        key.into_usize() < self.strings.len()
    }

    /// Gets the number of interned strings
    ///
    /// # Example
    ///
    /// ```rust
    /// use lasso_u16::Rodeo;
    ///
    /// // ThreadedRodeo is interchangeable for Rodeo here
    /// let mut rodeo = Rodeo::default();
    /// rodeo.get_or_intern("Documentation often has little hidden bits in it");
    ///
    /// let rodeo = rodeo.into_resolver();
    /// assert_eq!(rodeo.len(), 1);
    /// ```
    ///
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns `true` if there are no currently interned strings
    ///
    /// # Example
    ///
    /// ```rust
    /// use lasso_u16::Rodeo;
    ///
    /// // ThreadedRodeo is interchangeable for Rodeo here
    /// let rodeo = Rodeo::default();
    ///
    /// let rodeo = rodeo.into_resolver();
    /// assert!(rodeo.is_empty());
    /// ```
    ///
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the interned strings and their key values
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn iter(&self) -> Iter<'_, K> {
        Iter::from_resolver(self)
    }

    /// Returns an iterator over the interned strings
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn strings(&self) -> Strings<'_, K> {
        Strings::from_resolver(self)
    }
}

unsafe impl<K: Send> Send for RodeoResolver<K> {}
unsafe impl<K: Sync> Sync for RodeoResolver<K> {}

impl<'a, K: Key> IntoIterator for &'a RodeoResolver<K> {
    type Item = (K, &'a [u16]);
    type IntoIter = Iter<'a, K>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Key> Index<K> for RodeoResolver<K> {
    type Output = [u16];

    #[cfg_attr(feature = "inline-more", inline)]
    fn index(&self, idx: K) -> &Self::Output {
        self.resolve(&idx)
    }
}

impl<K> Eq for RodeoResolver<K> {}

impl<K> PartialEq<Self> for RodeoResolver<K> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn eq(&self, other: &Self) -> bool {
        self.strings == other.strings
    }
}

impl<K, S> PartialEq<RodeoReader<K, S>> for RodeoResolver<K> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn eq(&self, other: &RodeoReader<K, S>) -> bool {
        self.strings == other.strings
    }
}

impl<K, S> PartialEq<Rodeo<K, S>> for RodeoResolver<K> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn eq(&self, other: &Rodeo<K, S>) -> bool {
        self.strings == other.strings
    }
}

compile! {
    if #[feature = "serialize"] {
        use crate::Capacity;
        use core::num::NonZeroUsize;
        use serde::{
            de::{Deserialize, Deserializer},
            ser::{Serialize, Serializer},
        };
    }
}

#[cfg(feature = "serialize")]
impl<K> Serialize for RodeoResolver<K> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize all of self as a `Vec<String>`
        self.strings.serialize(serializer)
    }
}

#[cfg(feature = "serialize")]
impl<'de, K: Key> Deserialize<'de> for RodeoResolver<K> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vector: Vec<String> = Vec::deserialize(deserializer)?;
        let capacity = {
            let total_bytes = vector.iter().map(|s| s.len()).sum::<usize>();
            let total_bytes =
                NonZeroUsize::new(total_bytes).unwrap_or_else(|| Capacity::default().bytes());

            Capacity::new(vector.len(), total_bytes)
        };

        let mut strings = Vec::with_capacity(capacity.strings);
        let mut arena = Arena::new(capacity.bytes, usize::max_value())
            .expect("failed to allocate memory for interner");

        for string in vector {
            let allocated = unsafe {
                arena
                    .store_str(&string)
                    .expect("failed to allocate enough memory")
            };

            strings.push(allocated);
        }

        Ok(Self {
            strings,
            __arena: arena,
            __key: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    mod single_threaded {
        #[cfg(feature = "serialize")]
        use crate::RodeoResolver;
        use crate::{Key, Rodeo, Spur};

        #[test]
        fn resolve() {
            let mut rodeo = Rodeo::default();
            let key = rodeo.get_or_intern(&[0x41]);

            let resolver = rodeo.into_resolver();
            assert_eq!(&[0x41], resolver.resolve(&key));
        }

        #[test]
        #[should_panic]
        #[cfg(not(miri))]
        fn resolve_out_of_bounds() {
            let resolver = Rodeo::default().into_resolver();
            resolver.resolve(&Spur::try_from_usize(10).unwrap());
        }

        #[test]
        fn try_resolve() {
            let mut rodeo = Rodeo::default();
            let key = rodeo.get_or_intern(&[0x41]);

            let resolver = rodeo.into_resolver();
            let slice: &[u16] = &[0x41];
            assert_eq!(Some(slice), resolver.try_resolve(&key));
            assert_eq!(
                None,
                resolver.try_resolve(&Spur::try_from_usize(10).unwrap())
            );
        }

        #[test]
        fn resolve_unchecked() {
            let mut rodeo = Rodeo::default();
            let a = rodeo.get_or_intern(&[0x41]);

            let resolver = rodeo.into_resolver();
            unsafe {
                assert_eq!(&[0x41], resolver.resolve_unchecked(&a));
            }
        }

        #[test]
        fn len() {
            let mut rodeo = Rodeo::default();
            rodeo.get_or_intern(&[0x41]);
            rodeo.get_or_intern(&[0x42]);
            rodeo.get_or_intern(&[0x43]);

            let resolver = rodeo.into_resolver();
            assert_eq!(resolver.len(), 3);
        }

        #[test]
        fn empty() {
            let rodeo = Rodeo::default();
            let read_only = rodeo.into_resolver();

            assert!(read_only.is_empty());
        }

        #[test]
        fn drops() {
            let rodeo = Rodeo::default();
            let _ = rodeo.into_resolver();
        }

        #[test]
        fn iter() {
            let mut rodeo = Rodeo::default();
            let a = rodeo.get_or_intern(&[0x61]);
            let b = rodeo.get_or_intern(&[0x62]);
            let c = rodeo.get_or_intern(&[0x63]);

            let resolver = rodeo.into_resolver();
            let mut iter = resolver.iter();

            let slice: &[u16] = &[0x61];
            assert_eq!(Some((a, slice)), iter.next());
            let slice: &[u16] = &[0x62];
            assert_eq!(Some((b, slice)), iter.next());
            let slice: &[u16] = &[0x63];
            assert_eq!(Some((c, slice)), iter.next());
            assert_eq!(None, iter.next());
        }

        #[test]
        fn strings() {
            let mut rodeo = Rodeo::default();
            rodeo.get_or_intern(&[0x61]);
            rodeo.get_or_intern(&[0x62]);
            rodeo.get_or_intern(&[0x63]);

            let resolver = rodeo.into_resolver();
            let mut iter = resolver.strings();

            let left: Option<&[u16]> = Some(&[0x61]);
            assert_eq!(left, iter.next());
            let left: Option<&[u16]> = Some(&[0x62]);
            assert_eq!(left, iter.next());
            let left: Option<&[u16]> = Some(&[0x63]);
            assert_eq!(left, iter.next());
            assert_eq!(None, iter.next());
        }

        #[test]
        #[cfg(not(feature = "no-std"))]
        fn debug() {
            let resolver = Rodeo::default().into_resolver();
            println!("{:?}", resolver);
        }

        #[test]
        fn contains_key() {
            let mut rodeo = Rodeo::default();
            let key = rodeo.get_or_intern(&[]);
            let resolver = rodeo.into_resolver();

            assert!(resolver.contains_key(&key));
            assert!(!resolver.contains_key(&Spur::try_from_usize(10000).unwrap()));
        }

        #[test]
        fn into_iterator() {
            let rodeo = [&[0x61], &[0x62], &[0x63], &[0x64], &[0x65]]
                .iter()
                .collect::<Rodeo>()
                .into_resolver();

            for ((key, string), (expected_key, expected_string)) in rodeo.into_iter().zip(
                [
                    (0usize, &[0x61]),
                    (1, &[0x62]),
                    (2, &[0x63]),
                    (3, &[0x64]),
                    (4, &[0x65]),
                ]
                .iter()
                .copied(),
            ) {
                assert_eq!(key, Spur::try_from_usize(expected_key).unwrap());
                assert_eq!(string, expected_string);
            }
        }

        #[test]
        fn index() {
            let mut rodeo = Rodeo::default();
            let key = rodeo.get_or_intern(&[0x41]);

            let resolver = rodeo.into_resolver();
            assert_eq!(&[0x41], &resolver[key]);
        }

        #[test]
        #[cfg(feature = "serialize")]
        fn empty_serialize() {
            let rodeo = Rodeo::default().into_resolver();

            let ser = serde_json::to_string(&rodeo).unwrap();
            let ser2 = serde_json::to_string(&rodeo).unwrap();
            assert_eq!(ser, ser2);

            let deser: RodeoResolver = serde_json::from_str(&ser).unwrap();
            assert!(deser.is_empty());
            let deser2: RodeoResolver = serde_json::from_str(&ser2).unwrap();
            assert!(deser2.is_empty());
        }

        #[test]
        #[cfg(feature = "serialize")]
        fn filled_serialize() {
            let mut rodeo = Rodeo::default();
            let a = rodeo.get_or_intern(&[0x61]);
            let b = rodeo.get_or_intern(&[0x62]);
            let c = rodeo.get_or_intern(&[0x63]);
            let d = rodeo.get_or_intern(&[0x64]);
            let rodeo = rodeo.into_resolver();

            let ser = serde_json::to_string(&rodeo).unwrap();
            let ser2 = serde_json::to_string(&rodeo).unwrap();
            assert_eq!(ser, ser2);

            let deser: RodeoResolver = serde_json::from_str(&ser).unwrap();
            let deser2: RodeoResolver = serde_json::from_str(&ser2).unwrap();

            for (((correct_key, correct_str), (key1, str1)), (key2, str2)) in
                [(a, &[0x61]), (b, &[0x62]), (c, &[0x63]), (d, &[0x64])]
                    .iter()
                    .copied()
                    .zip(&deser)
                    .zip(&deser2)
            {
                assert_eq!(correct_key, key1);
                assert_eq!(correct_key, key2);

                assert_eq!(correct_str, str1);
                assert_eq!(correct_str, str2);
            }
        }

        #[test]
        fn resolver_eq() {
            let a = Rodeo::default();
            let b = Rodeo::default();
            assert_eq!(a.into_resolver(), b.into_resolver());

            let mut a = Rodeo::default();
            a.get_or_intern(&[0x61]);
            a.get_or_intern(&[0x62]);
            a.get_or_intern(&[0x63]);
            let mut b = Rodeo::default();
            b.get_or_intern(&[0x61]);
            b.get_or_intern(&[0x62]);
            b.get_or_intern(&[0x63]);
            assert_eq!(a.into_resolver(), b.into_resolver());
        }

        #[test]
        fn reader_eq() {
            let a = Rodeo::default();
            let b = Rodeo::default();
            assert_eq!(a.into_reader(), b.into_resolver());

            let mut a = Rodeo::default();
            a.get_or_intern(&[0x61]);
            a.get_or_intern(&[0x62]);
            a.get_or_intern(&[0x63]);
            let mut b = Rodeo::default();
            b.get_or_intern(&[0x61]);
            b.get_or_intern(&[0x62]);
            b.get_or_intern(&[0x63]);
            assert_eq!(a.into_reader(), b.into_resolver());
        }
    }

    #[cfg(all(not(any(miri, feature = "no-std")), features = "multi-threaded"))]
    mod multi_threaded {
        use crate::{locks::Arc, ThreadedRodeo};
        use std::thread;

        #[test]
        fn resolve() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let resolver = rodeo.into_resolver();
            assert_eq!(&[0x41], resolver.resolve(&key));
        }

        #[test]
        fn try_resolve() {
            let mut rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let resolver = rodeo.into_resolver();
            assert_eq!(Some(&[0x41]), resolver.try_resolve(&key));
            assert_eq!(
                None,
                resolver.try_resolve(&Spur::try_from_usize(10).unwrap())
            );
        }

        #[test]
        #[cfg(not(miri))]
        fn try_resolve_threaded() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let resolver = Arc::new(rodeo.into_resolver());

            let moved = Arc::clone(&resolver);
            thread::spawn(move || {
                assert_eq!(Some(&[0x41]), resolver.try_resolve(&key));
                assert_eq!(
                    None,
                    resolver.try_resolve(&Spur::try_from_usize(10).unwrap())
                );
            });

            assert_eq!(Some(&[0x41]), resolver.try_resolve(&key));
            assert_eq!(
                None,
                resolver.try_resolve(&Spur::try_from_usize(10).unwrap())
            );
        }

        #[test]
        fn resolve_unchecked() {
            let mut rodeo = ThreadedRodeo::default();
            let a = rodeo.get_or_intern(&[0x41]);

            let resolver = rodeo.into_resolver();
            unsafe {
                assert_eq!(&[0x41], resolver.resolve_unchecked(&a));
            }
        }

        #[test]
        #[cfg(not(miri))]
        fn resolve_unchecked_threaded() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let resolver = Arc::new(rodeo.into_resolver());

            let moved = Arc::clone(&resolver);
            thread::spawn(move || unsafe {
                assert_eq!(&[0x41], moved.resolve_unchecked(&a));
            });

            unsafe {
                assert_eq!(&[0x41], resolver.resolve_unchecked(&a));
            }
        }

        #[test]
        #[cfg(not(miri))]
        fn resolve_threaded() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let resolver = Arc::new(rodeo.into_resolver());

            let moved = Arc::clone(&resolver);
            thread::spawn(move || {
                assert_eq!(&[0x41], moved.resolve(&key));
            });

            assert_eq!(&[0x41], resolver.resolve(&key));
        }

        #[test]
        fn len() {
            let rodeo = ThreadedRodeo::default();
            rodeo.intern(&[0x41]);
            rodeo.intern(&[0x42]);
            rodeo.intern(&[0x43]);

            let resolver = rodeo.into_resolver();
            assert_eq!(resolver.len(), 3);
        }

        #[test]
        fn empty() {
            let rodeo = ThreadedRodeo::default();
            let read_only = rodeo.into_resolver();

            assert!(read_only.is_empty());
        }

        #[test]
        fn iter() {
            let rodeo = ThreadedRodeo::default();
            let a = rodeo.get_or_intern(&[0x61]);
            let b = rodeo.get_or_intern(&[0x62]);
            let c = rodeo.get_or_intern(&[0x63]);

            let resolver = rodeo.into_resolver();
            let mut iter = resolver.iter();

            assert_eq!(Some((a, &[0x61])), iter.next());
            assert_eq!(Some((b, &[0x62])), iter.next());
            assert_eq!(Some((c, &[0x63])), iter.next());
            assert_eq!(None, iter.next());
        }

        #[test]
        fn strings() {
            let rodeo = ThreadedRodeo::default();
            rodeo.get_or_intern(&[0x61]);
            rodeo.get_or_intern(&[0x62]);
            rodeo.get_or_intern(&[0x63]);

            let resolver = rodeo.into_resolver();
            let mut iter = resolver.strings();

            assert_eq!(Some(&[0x61]), iter.next());
            assert_eq!(Some(&[0x62]), iter.next());
            assert_eq!(Some(&[0x63]), iter.next());
            assert_eq!(None, iter.next());
        }

        #[test]
        fn clone() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x54, 0x65, 0x73, 0x74]);

            let resolver_rodeo = rodeo.into_resolver();
            assert_eq!(&[0x54, 0x65, 0x73, 0x74], resolver_rodeo.resolve(&key));

            let cloned = resolver_rodeo.clone();
            assert_eq!(&[0x54, 0x65, 0x73, 0x74], cloned.resolve(&key));

            drop(resolver_rodeo);

            assert_eq!(&[0x54, 0x65, 0x73, 0x74], cloned.resolve(&key));
        }

        #[test]
        fn drops() {
            let rodeo = ThreadedRodeo::default();
            let _ = rodeo.into_resolver();
        }

        #[test]
        #[cfg(not(miri))]
        fn drop_threaded() {
            let rodeo = ThreadedRodeo::default();
            let resolver = Arc::new(rodeo.into_resolver());

            let moved = Arc::clone(&resolver);
            thread::spawn(move || {
                let _ = moved;
            });
        }

        #[test]
        #[cfg(not(feature = "no-std"))]
        fn debug() {
            let resolver = ThreadedRodeo::default().into_resolver();
            println!("{:?}", resolver);
        }

        #[test]
        fn contains_key() {
            let mut rodeo = ThreadedRodeo::default();
            let key = rodeo.get_or_intern(&[]);
            let resolver = rodeo.into_resolver();

            assert!(resolver.contains_key(&key));
            assert!(!resolver.contains_key(&Spur::try_from_usize(10000).unwrap()));
        }

        #[test]
        fn into_iterator() {
            let rodeo = [&[0x61], &[0x62], &[0x63], &[0x64], &[0x65]]
                .iter()
                .collect::<ThreadedRodeo>()
                .into_resolver();

            for ((key, string), (expected_key, expected_string)) in rodeo.into_iter().zip(
                [
                    (0usize, &[0x61]),
                    (1, &[0x62]),
                    (2, &[0x63]),
                    (3, &[0x64]),
                    (4, &[0x65]),
                ]
                .iter()
                .copied(),
            ) {
                assert_eq!(key, Spur::try_from_usize(expected_key).unwrap());
                assert_eq!(string, expected_string);
            }
        }

        #[test]
        fn index() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let resolver = rodeo.into_resolver();
            assert_eq!(&[0x41], &resolver[key]);
        }

        #[test]
        #[cfg(feature = "serialize")]
        fn empty_serialize() {
            let rodeo = ThreadedRodeo::default().into_resolver();

            let ser = serde_json::to_string(&rodeo).unwrap();
            let ser2 = serde_json::to_string(&rodeo).unwrap();
            assert_eq!(ser, ser2);

            let deser: RodeoResolver = serde_json::from_str(&ser).unwrap();
            assert!(deser.is_empty());
            let deser2: RodeoResolver = serde_json::from_str(&ser2).unwrap();
            assert!(deser2.is_empty());
        }

        #[test]
        #[cfg(feature = "serialize")]
        fn filled_serialize() {
            let rodeo = ThreadedRodeo::default();
            let a = rodeo.get_or_intern(&[0x61]);
            let b = rodeo.get_or_intern(&[0x62]);
            let c = rodeo.get_or_intern(&[0x63]);
            let d = rodeo.get_or_intern(&[0x64]);
            let rodeo = rodeo.into_resolver();

            let ser = serde_json::to_string(&rodeo).unwrap();
            let ser2 = serde_json::to_string(&rodeo).unwrap();
            assert_eq!(ser, ser2);

            let deser: RodeoResolver = serde_json::from_str(&ser).unwrap();
            let deser2: RodeoResolver = serde_json::from_str(&ser2).unwrap();

            for (((correct_key, correct_str), (key1, str1)), (key2, str2)) in
                [(a, &[0x61]), (b, &[0x62]), (c, &[0x63]), (d, &[0x64])]
                    .iter()
                    .copied()
                    .zip(&deser)
                    .zip(&deser2)
            {
                assert_eq!(correct_key, key1);
                assert_eq!(correct_key, key2);

                assert_eq!(correct_str, str1);
                assert_eq!(correct_str, str2);
            }
        }
    }
}
