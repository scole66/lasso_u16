use crate::{
    arena::Arena,
    hasher::RandomState,
    keys::{Key, Spur},
    resolver::RodeoResolver,
    util::{Iter, Strings},
    Rodeo,
};
use core::{
    hash::{BuildHasher, Hash, Hasher},
    ops::Index,
};
use hashbrown::HashMap;

compile! {
    if #[feature = "no-std"] {
        use alloc::vec::Vec;
    }
}

/// A read-only view of a [`Rodeo`] or [`ThreadedRodeo`] that allows contention-free access to interned strings,
/// both key to string resolution and string to key lookups
///
/// The key and hasher types are the same as the `Rodeo` or `ThreadedRodeo` that created it, can be acquired with the
/// `into_reader` methods.
///
/// [`Rodeo`]: crate::Rodeo
/// [`ThreadedRodeo`]: crate::ThreadedRodeo
#[derive(Debug)]
pub struct RodeoReader<K = Spur, S = RandomState> {
    // The logic behind this arrangement is more heavily documented inside of
    // `Rodeo` itself
    map: HashMap<K, (), ()>,
    hasher: S,
    pub(crate) strings: Vec<&'static [u16]>,
    arena: Arena,
}

impl<K, S> RodeoReader<K, S> {
    /// Creates a new RodeoReader
    ///
    /// # Safety
    ///
    /// The references inside of `strings` must be absolutely unique, meaning
    /// that no other references to those strings exist
    ///
    pub(crate) unsafe fn new(
        map: HashMap<K, (), ()>,
        hasher: S,
        strings: Vec<&'static [u16]>,
        arena: Arena,
    ) -> Self {
        Self {
            map,
            hasher,
            strings,
            arena,
        }
    }

    /// Get the key value of a string, returning `None` if it doesn't exist
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
    /// let rodeo = rodeo.into_reader();
    /// assert_eq!(Some(key), rodeo.get("Strings of things with wings and dings"));
    ///
    /// assert_eq!(None, rodeo.get("This string isn't interned"));
    /// ```
    ///
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn get<T>(&self, val: T) -> Option<K>
    where
        T: AsRef<[u16]>,
        S: BuildHasher,
        K: Key,
    {
        let string_slice: &[u16] = val.as_ref();

        // Make a hash of the requested string
        let hash = {
            let mut state = self.hasher.build_hasher();
            string_slice.hash(&mut state);

            state.finish()
        };

        // Get the map's entry that the string should occupy
        let entry = self.map.raw_entry().from_hash(hash, |key| {
            // Safety: The index given by `key` will be in bounds of the strings vector
            let key_string: &[u16] = unsafe { index_unchecked!(self.strings, key.into_usize()) };

            // Compare the requested string against the key's string
            string_slice == key_string
        });

        entry.map(|(key, ())| *key)
    }

    /// Returns `true` if the given string has been interned
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
    /// let rodeo = rodeo.into_reader();
    /// assert!(rodeo.contains("Strings of things with wings and dings"));
    ///
    /// assert!(!rodeo.contains("This string isn't interned"));
    /// ```
    ///
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn contains<T>(&self, val: T) -> bool
    where
        T: AsRef<[u16]>,
        S: BuildHasher,
        K: Key,
    {
        self.get(val).is_some()
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
    /// let rodeo = rodeo.into_reader();
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

    /// Resolves a string by its key without bounds checks
    ///
    /// # Safety
    ///
    /// The key must be valid for the current Reader
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
    /// let rodeo = rodeo.into_reader();
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
    /// let rodeo = rodeo.into_reader();
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
        Iter::from_reader(self)
    }

    /// Returns an iterator over the interned strings
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn strings(&self) -> Strings<'_, K> {
        Strings::from_reader(self)
    }

    /// Consumes the current rodeo and makes it into a [`RodeoResolver`], allowing
    /// contention-free access from multiple threads with the lowest possible memory consumption
    ///
    /// # Example
    ///
    /// ```rust
    /// use lasso_u16::Rodeo;
    ///
    /// // ThreadedRodeo is interchangeable for Rodeo here
    /// let mut rodeo = Rodeo::default();
    /// let key = rodeo.get_or_intern("Appear weak when you are strong, and strong when you are weak.");
    /// let reader_rodeo = rodeo.into_reader();
    ///
    /// let resolver_rodeo = reader_rodeo.into_resolver();
    /// assert_eq!(
    ///     "Appear weak when you are strong, and strong when you are weak.",
    ///     resolver_rodeo.resolve(&key),
    /// );
    /// ```
    ///
    /// [`RodeoResolver`]: crate::RodeoResolver
    #[cfg_attr(feature = "inline-more", inline)]
    #[must_use]
    pub fn into_resolver(self) -> RodeoResolver<K> {
        let RodeoReader { strings, arena, .. } = self;

        // Safety: The current reader no longer contains references to the strings
        // in the vec given to RodeoResolver
        unsafe { RodeoResolver::new(strings, arena) }
    }
}

unsafe impl<K: Sync, S: Sync> Sync for RodeoReader<K, S> {}
unsafe impl<K: Send, S: Send> Send for RodeoReader<K, S> {}

impl<'a, K: Key, S> IntoIterator for &'a RodeoReader<K, S> {
    type Item = (K, &'a [u16]);
    type IntoIter = Iter<'a, K>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, S> Index<K> for RodeoReader<K, S>
where
    K: Key,
    S: BuildHasher,
{
    type Output = [u16];

    #[cfg_attr(feature = "inline-more", inline)]
    fn index(&self, idx: K) -> &Self::Output {
        self.resolve(&idx)
    }
}

impl<K, S> Eq for RodeoReader<K, S> {}

impl<K, S> PartialEq<Self> for RodeoReader<K, S> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn eq(&self, other: &Self) -> bool {
        self.strings == other.strings
    }
}

impl<K, S> PartialEq<RodeoResolver<K>> for RodeoReader<K, S> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn eq(&self, other: &RodeoResolver<K>) -> bool {
        self.strings == other.strings
    }
}

impl<K, S> PartialEq<Rodeo<K, S>> for RodeoReader<K, S> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn eq(&self, other: &Rodeo<K, S>) -> bool {
        self.strings == other.strings
    }
}

compile! {
    if #[feature = "serialize"] {
        use crate::Capacity;
        use core::num::NonZeroUsize;
        use hashbrown::hash_map::RawEntryMut;
        use serde::{
            de::{Deserialize, Deserializer},
            ser::{Serialize, Serializer},
        };
    }
}

#[cfg(feature = "serialize")]
impl<K, H> Serialize for RodeoReader<K, H> {
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
impl<'de, K: Key, S: BuildHasher + Default> Deserialize<'de> for RodeoReader<K, S> {
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

        let hasher: S = Default::default();
        let mut strings = Vec::with_capacity(capacity.strings);
        let mut map = HashMap::with_capacity_and_hasher(capacity.strings, ());
        let mut arena = Arena::new(capacity.bytes, usize::max_value())
            .expect("failed to allocate memory for interner");

        for (key, string) in vector.into_iter().enumerate() {
            let allocated = unsafe {
                arena
                    .store_str(&string)
                    .expect("failed to allocate enough memory")
            };

            let hash = {
                let mut state = hasher.build_hasher();
                allocated.hash(&mut state);

                state.finish()
            };

            // Get the map's entry that the string should occupy
            let entry = map.raw_entry_mut().from_hash(hash, |key: &K| {
                // Safety: The index given by `key` will be in bounds of the strings vector
                let key_string: &[u16] = unsafe { index_unchecked!(strings, key.into_usize()) };

                // Compare the requested string against the key's string
                allocated == key_string
            });

            match entry {
                RawEntryMut::Occupied(..) => {
                    debug_assert!(false, "re-interned a key while deserializing");
                }
                RawEntryMut::Vacant(entry) => {
                    // Create the key from the vec's index that the string will hold
                    let key =
                        K::try_from_usize(key).expect("failed to create key while deserializing");

                    // Push the allocated string to the strings vector
                    strings.push(allocated);

                    // Insert the key with the hash of the string that it points to, reusing the hash we made earlier
                    entry.insert_with_hasher(hash, key, (), |key| {
                        let key_string: &[u16] =
                            unsafe { index_unchecked!(strings, key.into_usize()) };

                        let mut state = hasher.build_hasher();
                        key_string.hash(&mut state);

                        state.finish()
                    });
                }
            }
        }

        Ok(Self {
            map,
            hasher,
            strings,
            arena,
        })
    }
}

#[cfg(test)]
mod tests {
    mod single_threaded {
        #[cfg(feature = "serialize")]
        use crate::RodeoReader;
        use crate::{Key, Rodeo, Spur};

        #[test]
        fn get() {
            let mut rodeo = Rodeo::default();
            let key = rodeo.get_or_intern(&[0x41]);

            let reader = rodeo.into_reader();
            assert_eq!(Some(key), reader.get(&[0x41]));

            assert!(reader.get(&[0x46]).is_none());
        }

        #[test]
        fn resolve() {
            let mut rodeo = Rodeo::default();
            let key = rodeo.get_or_intern(&[0x41]);

            let reader = rodeo.into_reader();
            assert_eq!(&[0x41], reader.resolve(&key));
        }

        #[test]
        #[should_panic]
        #[cfg(not(miri))]
        fn resolve_panics() {
            let reader = Rodeo::default().into_reader();
            reader.resolve(&Spur::try_from_usize(100).unwrap());
        }

        #[test]
        fn try_resolve() {
            let mut rodeo = Rodeo::default();
            let key = rodeo.get_or_intern(&[0x41]);

            let reader = rodeo.into_reader();
            let left: &[u16] = &[0x41];
            assert_eq!(Some(left), reader.try_resolve(&key));
            assert_eq!(
                None,
                reader.try_resolve(&Spur::try_from_usize(100).unwrap())
            );
        }

        #[test]
        fn resolve_unchecked() {
            let mut rodeo = Rodeo::default();
            let key = rodeo.get_or_intern(&[0x41]);

            let reader = rodeo.into_reader();
            unsafe {
                assert_eq!(&[0x41], reader.resolve_unchecked(&key));
            }
        }

        #[test]
        fn len() {
            let mut rodeo = Rodeo::default();
            rodeo.get_or_intern(&[0x41]);
            rodeo.get_or_intern(&[0x42]);
            rodeo.get_or_intern(&[0x43]);

            let reader = rodeo.into_reader();
            assert_eq!(reader.len(), 3);
        }

        #[test]
        fn empty() {
            let rodeo = Rodeo::default();
            let reader = rodeo.into_reader();

            assert!(reader.is_empty());
        }

        #[test]
        fn iter() {
            let mut rodeo = Rodeo::default();
            let a = rodeo.get_or_intern(&[0x61]);
            let b = rodeo.get_or_intern(&[0x62]);
            let c = rodeo.get_or_intern(&[0x63]);

            let resolver = rodeo.into_reader();
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

            let resolver = rodeo.into_reader();
            let mut iter = resolver.strings();

            let slice: &[u16] = &[0x61];
            assert_eq!(Some(slice), iter.next());
            let slice: &[u16] = &[0x62];
            assert_eq!(Some(slice), iter.next());
            let slice: &[u16] = &[0x63];
            assert_eq!(Some(slice), iter.next());
            assert_eq!(None, iter.next());
        }

        #[test]
        fn drops() {
            let rodeo = Rodeo::default();
            let _ = rodeo.into_reader();
        }

        #[test]
        fn into_resolver() {
            let mut rodeo = Rodeo::default();
            let key = rodeo.get_or_intern(&[0x41]);

            let resolver = rodeo.into_reader().into_resolver();
            assert_eq!(&[0x41], resolver.resolve(&key));
        }

        #[test]
        #[cfg(not(any(feature = "no-std", feature = "ahasher")))]
        fn debug() {
            let reader = Rodeo::default().into_reader();
            println!("{:?}", reader);
        }

        #[test]
        fn contains() {
            let mut rodeo = Rodeo::default();
            rodeo.get_or_intern(&[]);
            let resolver = rodeo.into_reader();

            assert!(resolver.contains(&[]));
            assert!(resolver.contains(&[]));
        }

        #[test]
        fn contains_key() {
            let mut rodeo = Rodeo::default();
            let key = rodeo.get_or_intern(&[]);
            let resolver = rodeo.into_reader();

            assert!(resolver.contains(&[]));
            assert!(resolver.contains_key(&key));
            assert!(!resolver.contains_key(&Spur::try_from_usize(10000).unwrap()));
        }

        #[test]
        fn into_iterator() {
            let rodeo = [&[0x61], &[0x62], &[0x63], &[0x64], &[0x65]]
                .iter()
                .collect::<Rodeo>()
                .into_reader();

            for ((key, string), (expected_key, expected_string)) in rodeo.into_iter().zip(
                [(0usize, &[0x61]), (1, &[0x62]), (2, &[0x63]), (3, &[0x64]), (4, &[0x65])]
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

            let reader = rodeo.into_reader();
            assert_eq!(&[0x41], &reader[key]);
        }

        #[test]
        #[cfg(feature = "serialize")]
        fn empty_serialize() {
            let rodeo = Rodeo::default().into_reader();

            let ser = serde_json::to_string(&rodeo).unwrap();
            let ser2 = serde_json::to_string(&rodeo).unwrap();
            assert_eq!(ser, ser2);

            let deser: RodeoReader = serde_json::from_str(&ser).unwrap();
            assert!(deser.is_empty());
            let deser2: RodeoReader = serde_json::from_str(&ser2).unwrap();
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
            let rodeo = rodeo.into_reader();

            let ser = serde_json::to_string(&rodeo).unwrap();
            let ser2 = serde_json::to_string(&rodeo).unwrap();
            assert_eq!(ser, ser2);

            let deser: RodeoReader = serde_json::from_str(&ser).unwrap();
            let deser2: RodeoReader = serde_json::from_str(&ser2).unwrap();

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
        fn reader_eq() {
            let a = Rodeo::default();
            let b = Rodeo::default();
            assert_eq!(a.into_reader(), b.into_reader());

            let mut a = Rodeo::default();
            a.get_or_intern(&[0x61]);
            a.get_or_intern(&[0x62]);
            a.get_or_intern(&[0x63]);
            let mut b = Rodeo::default();
            b.get_or_intern(&[0x61]);
            b.get_or_intern(&[0x62]);
            b.get_or_intern(&[0x63]);
            assert_eq!(a.into_reader(), b.into_reader());
        }

        #[test]
        fn resolver_eq() {
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
        use crate::{locks::Arc, RodeoReader, ThreadedRodeo};
        use std::thread;

        #[test]
        fn get() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let reader = rodeo.into_reader();
            assert_eq!(Some(key), reader.get(&[0x41]));

            assert!(reader.get(&[0x46]).is_none());
        }

        #[test]
        #[cfg(not(miri))]
        fn get_threaded() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let reader = Arc::new(rodeo.into_reader());

            let moved = Arc::clone(&reader);
            thread::spawn(move || {
                assert_eq!(Some(key), moved.get(&[0x41]));
                assert!(moved.get(&[0x46]).is_none());
            });

            assert_eq!(Some(key), reader.get(&[0x41]));
            assert!(reader.get(&[0x46]).is_none());
        }

        #[test]
        fn resolve() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let reader = rodeo.into_reader();
            assert_eq!(&[0x41], reader.resolve(&key));
        }

        #[test]
        #[cfg(not(miri))]
        fn resolve_threaded() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let reader = Arc::new(rodeo.into_reader());

            let moved = Arc::clone(&reader);
            thread::spawn(move || {
                assert_eq!(&[0x41], moved.resolve(&key));
            });

            assert_eq!(&[0x41], reader.resolve(&key));
        }

        #[test]
        fn len() {
            let rodeo = ThreadedRodeo::default();
            rodeo.intern(&[0x41]);
            rodeo.intern(&[0x42]);
            rodeo.intern(&[0x43]);

            let reader = rodeo.into_reader();
            assert_eq!(reader.len(), 3);
        }

        #[test]
        fn empty() {
            let rodeo = ThreadedRodeo::default();
            let reader = rodeo.into_reader();

            assert!(reader.is_empty());
        }

        #[test]
        fn clone() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x54, 0x65, 0x73, 0x74]);

            let reader_rodeo = rodeo.into_reader();
            assert_eq!(&[0x54, 0x65, 0x73, 0x74], reader_rodeo.resolve(&key));

            let cloned = reader_rodeo.clone();
            assert_eq!(&[0x54, 0x65, 0x73, 0x74], cloned.resolve(&key));

            drop(reader_rodeo);

            assert_eq!(&[0x54, 0x65, 0x73, 0x74], cloned.resolve(&key));
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
        fn drops() {
            let rodeo = ThreadedRodeo::default();
            let _ = rodeo.into_reader();
        }

        #[test]
        #[cfg(not(miri))]
        fn drop_threaded() {
            let rodeo = ThreadedRodeo::default();
            let reader = Arc::new(rodeo.into_reader());

            let moved = Arc::clone(&reader);
            thread::spawn(move || {
                let _ = moved;
            });
        }

        #[test]
        fn into_resolver() {
            let rodeo = ThreadedRodeo::default();
            let key = rodeo.intern(&[0x41]);

            let resolver = rodeo.into_reader().into_resolver();
            assert_eq!(&[0x41], resolver.resolve(&key));
        }

        #[test]
        #[cfg(not(feature = "no-std"))]
        fn debug() {
            let reader = ThreadedRodeo::default().into_reader();
            println!("{:?}", reader);
        }

        #[test]
        fn contains() {
            let mut rodeo = ThreadedRodeo::default();
            rodeo.get_or_intern(&[]);
            let resolver = rodeo.into_reader();

            assert!(resolver.contains(&[]));
            assert!(resolver.contains(&[]));
        }

        #[test]
        fn contains_key() {
            let mut rodeo = ThreadedRodeo::default();
            let key = rodeo.get_or_intern(&[]);
            let resolver = rodeo.into_reader();

            assert!(resolver.contains(&[]));
            assert!(resolver.contains_key(&key));
            assert!(!resolver.contains_key(&Spur::try_from_usize(10000).unwrap()));
        }

        #[test]
        fn into_iterator() {
            let rodeo = [&[0x61], &[0x62], &[0x63], &[0x64], &[0x65]]
                .iter()
                .collect::<ThreadedRodeo>()
                .into_reader();

            for ((key, string), (expected_key, expected_string)) in rodeo.into_iter().zip(
                [(0usize, &[0x61]), (1, &[0x62]), (2, &[0x63]), (3, &[0x64]), (4, &[0x65])]
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

            let reader = rodeo.into_reader();
            assert_eq!(&[0x41], &reader[key]);
        }

        #[test]
        #[cfg(feature = "serialize")]
        fn empty_serialize() {
            let rodeo = ThreadedRodeo::default().into_reader();

            let ser = serde_json::to_string(&rodeo).unwrap();
            let ser2 = serde_json::to_string(&rodeo).unwrap();
            assert_eq!(ser, ser2);

            let deser: RodeoReader = serde_json::from_str(&ser).unwrap();
            assert!(deser.is_empty());
            let deser2: RodeoReader = serde_json::from_str(&ser2).unwrap();
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
            let rodeo = rodeo.into_reader();

            let ser = serde_json::to_string(&rodeo).unwrap();
            let ser2 = serde_json::to_string(&rodeo).unwrap();
            assert_eq!(ser, ser2);

            let deser: RodeoReader = serde_json::from_str(&ser).unwrap();
            let deser2: RodeoReader = serde_json::from_str(&ser2).unwrap();

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
