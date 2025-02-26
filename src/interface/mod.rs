mod boxed;
mod rodeo;
mod rodeo_reader;
mod rodeo_resolver;
mod tests;
mod threaded_ref;
mod threaded_rodeo;

use crate::{Key, LassoResult, Spur};
#[cfg(feature = "no-std")]
use alloc::boxed::Box;

/// A generic interface over any underlying interner, allowing storing and accessing
/// interned strings
///
/// Note that because single-threaded [`Rodeo`](crate::Rodeo)s require mutable access to use, this
/// trait does so as well. For use with [`ThreadedRodeo`](crate::ThreadedRodeo), the trait is
/// implemented for `&ThreadedRodeo` as well to allow access through shared references.
pub trait Interner<K = Spur>: Reader<K> + Resolver<K> {
    /// Get the key for a string, interning it if it does not yet exist
    ///
    /// # Panics
    ///
    /// Panics if the key's [`try_from_usize`](Key::try_from_usize) function fails. With the default
    /// keys, this means that you've interned more strings than it can handle. (For [`Spur`] this
    /// means that `u32::MAX - 1` unique strings were interned)
    ///
    fn get_or_intern(&mut self, val: &[u16]) -> K;

    /// Get the key for a string, interning it if it does not yet exist
    fn try_get_or_intern(&mut self, val: &[u16]) -> LassoResult<K>;

    /// Get the key for a static string, interning it if it does not yet exist
    ///
    /// This will not reallocate or copy the given string
    ///
    /// # Panics
    ///
    /// Panics if the key's [`try_from_usize`](Key::try_from_usize) function fails. With the default
    /// keys, this means that you've interned more strings than it can handle. (For [`Spur`] this
    /// means that `u32::MAX - 1` unique strings were interned)
    ///
    fn get_or_intern_static(&mut self, val: &'static [u16]) -> K;

    /// Get the key for a static string, interning it if it does not yet exist
    ///
    /// This will not reallocate or copy the given string
    fn try_get_or_intern_static(&mut self, val: &'static [u16]) -> LassoResult<K>;
}

/// A generic interface over interners that can be turned into both a [`Reader`] and a [`Resolver`]
/// directly.
pub trait IntoReaderAndResolver<K = Spur>: IntoReader<K> + IntoResolver<K>
where
    K: Key,
{
}

/// A generic interface over interners that can be turned into a [`Reader`].
pub trait IntoReader<K = Spur>: Interner<K>
where
    K: Key,
{
    /// The type of [`Reader`] the interner will be converted into
    type Reader: Reader<K>;

    /// Consumes the current [`Interner`] and converts it into a [`Reader`] to allow
    /// contention-free access of the interner from multiple threads
    #[must_use]
    fn into_reader(self) -> Self::Reader
    where
        Self: 'static;

    /// An implementation detail to allow calling [`Interner::into_reader()`] on dynamically
    /// dispatched interners
    #[doc(hidden)]
    #[must_use]
    fn into_reader_boxed(self: Box<Self>) -> Self::Reader
    where
        Self: 'static;
}

/// A generic interface that allows using any underlying interner for
/// both its reading and resolution capabilities, allowing both
/// `[u16] -> key` and `key -> [u16]` lookups
pub trait Reader<K = Spur>: Resolver<K> {
    /// Get a key for the given string value if it exists
    fn get(&self, val: &[u16]) -> Option<K>;

    /// Returns `true` if the current interner contains the given string
    fn contains(&self, val: &[u16]) -> bool;
}

/// A generic interface over [`Reader`]s that can be turned into a [`Resolver`].
pub trait IntoResolver<K = Spur>: Reader<K>
where
    K: Key,
{
    /// The type of [`Resolver`] the reader will be converted into
    type Resolver: Resolver<K>;

    /// Consumes the current [`Reader`] and makes it into a [`Resolver`], allowing
    /// contention-free access from multiple threads with the lowest possible memory consumption
    #[must_use]
    fn into_resolver(self) -> Self::Resolver
    where
        Self: 'static;

    /// An implementation detail to allow calling [`Reader::into_resolver()`] on dynamically
    /// dispatched readers
    #[doc(hidden)]
    #[must_use]
    fn into_resolver_boxed(self: Box<Self>) -> Self::Resolver
    where
        Self: 'static;
}

/// A generic interface that allows using any underlying interner only
/// for its resolution capabilities, allowing only `key -> [u16]` lookups
pub trait Resolver<K = Spur> {
    /// Resolves the given key into a string
    ///
    /// # Panics
    ///
    /// Panics if the key is not contained in the current [`Resolver`]
    ///
    fn resolve<'a>(&'a self, key: &K) -> &'a [u16];

    /// Attempts to resolve the given key into a string, returning `None`
    /// if it cannot be found
    fn try_resolve<'a>(&'a self, key: &K) -> Option<&'a [u16]>;

    /// Resolves a string by its key without preforming bounds checks
    ///
    /// # Safety
    ///
    /// The key must be valid for the current [`Resolver`]
    ///
    unsafe fn resolve_unchecked<'a>(&'a self, key: &K) -> &'a [u16];

    /// Returns `true` if the current interner contains the given key
    fn contains_key(&self, key: &K) -> bool;

    /// Gets the number of currently interned strings
    fn len(&self) -> usize;

    /// Returns `true` if there are no currently interned strings
    #[cfg_attr(feature = "inline-more", inline)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
