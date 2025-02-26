//! Implementations of [`Interner`], [`Reader`] and [`Resolver`] for [`Rodeo`]

use crate::*;
#[cfg(feature = "no-std")]
use alloc::boxed::Box;
use core::hash::BuildHasher;
use interface::IntoReaderAndResolver;

impl<K, S> Interner<K> for Rodeo<K, S>
where
    K: Key,
    S: BuildHasher,
{
    #[cfg_attr(feature = "inline-more", inline)]
    fn get_or_intern(&mut self, val: &[u16]) -> K {
        self.get_or_intern(val)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn try_get_or_intern(&mut self, val: &[u16]) -> LassoResult<K> {
        self.try_get_or_intern(val)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn get_or_intern_static(&mut self, val: &'static [u16]) -> K {
        self.get_or_intern_static(val)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn try_get_or_intern_static(&mut self, val: &'static [u16]) -> LassoResult<K> {
        self.try_get_or_intern_static(val)
    }
}

impl<K, S> IntoReaderAndResolver<K> for Rodeo<K, S>
where
    K: Key,
    S: BuildHasher,
{
}

impl<K, S> IntoReader<K> for Rodeo<K, S>
where
    K: Key,
    S: BuildHasher,
{
    type Reader = RodeoReader<K, S>;

    #[cfg_attr(feature = "inline-more", inline)]
    #[must_use]
    fn into_reader(self) -> Self::Reader
    where
        Self: 'static,
    {
        self.into_reader()
    }

    #[cfg_attr(feature = "inline-more", inline)]
    #[must_use]
    fn into_reader_boxed(self: Box<Self>) -> Self::Reader
    where
        Self: 'static,
    {
        Rodeo::into_reader(*self)
    }
}

impl<K, S> Reader<K> for Rodeo<K, S>
where
    K: Key,
    S: BuildHasher,
{
    #[cfg_attr(feature = "inline-more", inline)]
    fn get(&self, val: &[u16]) -> Option<K> {
        self.get(val)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn contains(&self, val: &[u16]) -> bool {
        self.contains(val)
    }
}

impl<K, S> IntoResolver<K> for Rodeo<K, S>
where
    K: Key,
    S: BuildHasher,
{
    type Resolver = RodeoResolver<K>;

    #[cfg_attr(feature = "inline-more", inline)]
    #[must_use]
    fn into_resolver(self) -> Self::Resolver
    where
        Self: 'static,
    {
        self.into_resolver()
    }

    #[cfg_attr(feature = "inline-more", inline)]
    #[must_use]
    fn into_resolver_boxed(self: Box<Self>) -> Self::Resolver
    where
        Self: 'static,
    {
        Rodeo::into_resolver(*self)
    }
}

impl<K, S> Resolver<K> for Rodeo<K, S>
where
    K: Key,
{
    #[cfg_attr(feature = "inline-more", inline)]
    fn resolve<'a>(&'a self, key: &K) -> &'a [u16] {
        self.resolve(key)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn try_resolve<'a>(&'a self, key: &K) -> Option<&'a [u16]> {
        self.try_resolve(key)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    unsafe fn resolve_unchecked<'a>(&'a self, key: &K) -> &'a [u16] {
        self.resolve_unchecked(key)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn contains_key(&self, key: &K) -> bool {
        self.contains_key(key)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn len(&self) -> usize {
        self.len()
    }
}
