/* Hash Tables Implementation.
 *
 * This file implements in memory dictionaries with insert/del/replace/find/
 * get-random-element operations. The dictionary is implemented as a Hash Array
 * Mapped Trie (HAMT). See the source code for more information... :)
 *
 * Copyright (c) 2006-2012, Salvatore Sanfilippo <antirez at gmail dot com>
 * Copyright (c) 2021, Viktor Söderqvist <viktor.soderqvist@est.tech>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "fmacros.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <sys/time.h>
#include <math.h>

#include "dict.h"
#include "zmalloc.h"
#include "redisassert.h"

/* Our dict is implemented as a Hash Array Mapped Trie (HAMT). The hash (result
 * of a hash function) of each key is stored in a a tree structure where each
 * node has up to 32 children. With this large branching factor, the tree never
 * becomes very deep (6-7 levels). The time complexity is O(log32 n), which is
 * very close to constant.
 *
 * This is our HAMT structure:
 *
 * +--------------------+
 * | dict               |
 * |--------------------|   +--------------------+   +--------------------+
 * | children | bitmap ---->| children | bitmap ---->| key      | value   |
 * | size               |   | key      | value   |   | ...      | ...     |
 * | dictType           |   | key      | value   |   +--------------------+
 * | privdata           |   | key      | value   |
 * +--------------------+   | key      | value   |   +--------------------+
 *                          | children | bitmap ---->| key      | value   |
 *                          | key      | value   |   | children | bitmap --->..
 *                          | ...      | ...     |   | value    | key     |
 *                          +--------------------+   | ...      | ...     |
 *                                                   +--------------------+
 *
 * In each level, 5 bits of the hash is used as an index into the child nodes.
 * The bitmap indicates which of the children exist (1) and which don't (0).
 *
 * hash(key) = 11010 10101 11001 01001 010101 10111 ...
 *             ^^^^^
 *              26
 *
 * bitmap    = 00100101 00000010 00010100 01100001
 * (level 0)        ^
 *                  bit 26 is set, so there is a child node or leaf
 *
 * The children array is compact and its length is equal to the number of 1s in
 * the bitmap. In the bitmap above, our bit is the 2nd 1-bit from the left, so
 * we look at the 2nd child.
 *
 * children  = +-------------------+
 *             | key      | value  |
 *             | key      | value  | <-- The 2nd child is a key-value entry. If
 *             | children | bitmap |     the key matches our key, then it's our
 *             | ...      | ...    |     key. Otherwise, the dict doesn't
 *             +-------------------+     contain our key.
 *
 * If, instead, our child is a sub node (a children-bitmap pair), we use the
 * next 5 bits from the hash to index into the next level children.
 *
 * The HAMT is described in Phil Bagwell (2000). Ideal Hash Trees.
 * https://infoscience.epfl.ch/record/64398/files/idealhashtrees.pdf
 *
 * Until 2021, the dictionary was implemented as a hash table with incremental
 * rehashing using two hash tables while rehashing and resized in powers of two.
 * Despite the incremental rehashing, allocating and freeing very large
 * continous memory is expensive. The memory usage was also higher than for the
 * HAMT (TO BE VERIFIED). */

/* -------------------------- globals --------------------------------------- */

/* Using dictEnableResize() / dictDisableResize() we make possible to
 * enable/disable resizing of the hash table as needed. This is very important
 * for Redis, as we use copy-on-write and don't want to move too much memory
 * around when there is a child performing saving operations.
 *
 * Note that even when dict_can_resize is set to 0, not all resizes are
 * prevented: a hash table is still allowed to grow if the ratio between
 * the number of elements and the buckets > dict_force_resize_ratio. */
static int dict_can_resize = 1;
/* static unsigned int dict_force_resize_ratio = 5; */

/* -------------------------- private prototypes ---------------------------- */

/* -------------------------- hash functions -------------------------------- */

static uint8_t dict_hash_function_seed[16];

void dictSetHashFunctionSeed(uint8_t *seed) {
    memcpy(dict_hash_function_seed,seed,sizeof(dict_hash_function_seed));
}

uint8_t *dictGetHashFunctionSeed(void) {
    return dict_hash_function_seed;
}

/* The default hashing function uses SipHash implementation
 * in siphash.c. */

uint64_t siphash(const uint8_t *in, const size_t inlen, const uint8_t *k);
uint64_t siphash_nocase(const uint8_t *in, const size_t inlen, const uint8_t *k);

uint64_t dictGenHashFunction(const void *key, int len) {
    return siphash(key,len,dict_hash_function_seed);
}

uint64_t dictGenCaseHashFunction(const unsigned char *buf, int len) {
    return siphash_nocase(buf,len,dict_hash_function_seed);
}

/* ---------------- Macros and functions for bit manipulation --------------- */

/* Using builtin popcount CPU instruction is ~4 times faster than the
 * bit-twiddling emulation below. The builtin requires GCC >= 3.4.0 or a recent
 * Clang. To utilize the CPU popcount instruction, compile with -msse4.2 or use
 * #pragma GCC target ("sse4.2"). */
#ifdef USE_BUILTIN_POPCOUNT
# define dict_popcount(x)  __builtin_popcount((unsigned int)(x))
#else
/* Population count (the number of 1s in a binary number), SWAR algorithm. */
int dict_popcount(uint32_t x) {
    x -= ((x >> 1) & 0x55555555);
    x  = ((x >> 2) & 0x33333333) + (x & 0x33333333);
    x  = ((x >> 4) & 0x0f0f0f0f) + (x & 0x0f0f0f0f);
    x += (x >> 8);
    return (x + (x >> 16)) & 0x3f;
}
#endif

/* The dictEntry.key and dictSubNode.children share space in the union. The two
   least significant bits determine if it's an entry or a sub-node. Keys are
   aligned on at least 4 bytes (2-bit LSB pattern 00), except sds strings, which
   have an odd-sized (1, 3, 5 or 9 bytes) header before the pointer, so sds
   pointers have LSB = 1. We use the 2-bit pattern 10 to tag/mask the children
   pointers. */
#define is_childptr(ptr) (((intptr_t)(ptr) & 3) == 2)
#define mask_childptr(ptr) ((union dictNode*)(((intptr_t)(ptr)) | 2))
#define unmask_childptr(ptr) ((union dictNode*)(((intptr_t)(ptr)) & ~2))
#define is_entry(ptr) (!is_childptr((ptr)->sub.children))

/* The 5 bits used at level N to index into the children. */
#define bitindex_at_level(h, lvl) (((h) >> (5 * (lvl))) & 0x1f)

/* Sets the 5 bits corresponding at a given level to bitindex (0-31) */
static void setBitindexAtLevel(uint64_t *path, int level, int bitindex) {
    *path &= ~(0x1f << (5 * level)); /* clear bits */
    *path |= (bitindex << (5 * level)); /* set bits */
}

/* If we'd want to use the MSB of the hash first...
 * For level 0, it's the first 5 bits of 64. For level N, it's the 5 bits
 * after skipping the first N * 5 bits. */
/* #define bitindex_at_level(h, lvl) (((h) >> (64 - 5 - (5 * lvl))) & 0x1f) */

/* ----------------------------- API implementation ------------------------- */

/* Create a new hash table */
dict *dictCreate(dictType *type,
        void *privDataPtr)
{
    dict *d = zmalloc(sizeof(*d));
    d->type = type;
    d->privdata = privDataPtr;
    d->size = 0;
    return d;
}

/* Performs N steps of incremental rehashing. Returns 1 if there are still
 * keys to move from the old to the new hash table, otherwise 0 is returned.
 */
int dictRehash(dict *d, int n) {
    DICT_NOTUSED(d);
    DICT_NOTUSED(n);
    return 0;
}

long long timeInMilliseconds(void) {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000)+(tv.tv_usec/1000);
}

/* Rehash in ms+"delta" milliseconds. The value of "delta" is larger 
 * than 0, and is smaller than 1 in most cases. The exact upper bound 
 * depends on the running time of dictRehash(d,100).*/
int dictRehashMilliseconds(dict *d, int ms) {
    DICT_NOTUSED(d);
    DICT_NOTUSED(ms);
    return 0;
}

/* Add an element to the target hash table */
int dictAdd(dict *d, void *key, void *val)
{
    dictEntry *entry = dictAddRaw(d,key,NULL);

    if (!entry) return DICT_ERR;
    dictSetVal(d, entry, val);
    return DICT_OK;
}

/* Low level add or find:
 * This function adds the entry but instead of setting a value returns the
 * dictEntry structure to the user, that will make sure to fill the value
 * field as they wish.
 *
 * This function is also directly exposed to the user API to be called
 * mainly in order to store non-pointers inside the hash value, example:
 *
 * entry = dictAddRaw(dict,mykey,NULL);
 * if (entry != NULL) dictSetSignedIntegerVal(entry,1000);
 *
 * Return values:
 *
 * If key already exists NULL is returned, and "*existing" is populated
 * with the existing entry if existing is not NULL.
 *
 * If key was added, the hash entry is returned to be manipulated by the caller.
 */
dictEntry *dictAddRaw(dict *d, void *key, dictEntry **existing) {
    /* Keys are not allowed to have the 2-bit LSB pattern 10, since this is how
       we tag internal children pointers. */
    assert(!is_childptr(key));

    if (d->size == 0) {
        d->root.entry.key = key;
        d->size = 1;
        return &d->root.entry;
    }

    uint64_t hash = dictHashKey(d, key);
    int level = 0;
    union dictNode *node = &d->root;
    for (;;) {
        if (is_childptr(node->sub.children)) {
            /* It's an internal node. */
            union dictNode *children = unmask_childptr(node->sub.children);
            uint32_t bitmap = node->sub.bitmap;
            /* Use 5 bits of the hash as an index into one of 32 possible
               children. The child exists if the bit at bitindex is set. */
            int bitindex = bitindex_at_level(hash, level);
            uint32_t shifted = bitmap >> bitindex;
            if (shifted & 1) {
                /* Child exists. The position of the child is the number of
                 * 1-bits to the left of this bit in the bitmap. */
                int childindex = dict_popcount(shifted >> 1);
                node = &children[childindex];
                if (level * 5 >= 64) {
                    /* We've used up all hash bits. Use 2ndary hash function. */
                    assert(false); /* TODO */
                }
                level++;
            } else {
                /* Child doesn't exist. Let's make space for our entry here. */
                int childindex = dict_popcount(shifted >> 1);
                int numchildren = dict_popcount(bitmap);
                int numafter = numchildren - childindex;
                children = zrealloc(children,
                                    sizeof(union dictNode) * ++numchildren);
                memmove(&children[childindex + 1], &children[childindex],
                        sizeof(union dictNode) * numafter);
                children[childindex].entry.key = key;
                node->sub.children = mask_childptr(children);
                node->sub.bitmap |= (1 << bitindex);
                d->size++;
                return &children[childindex].entry;
            }
        } else {
            /* It's a leaf, i.e. a dictEntry. */
            if (dictCompareKeys(d, key, node->entry.key)) {
                *existing = &node->entry;
                return NULL;
            }
            /* There's another entry at this position. We need to wrap this one
               in a sub node (and then add our entry in there). Copy the
               existing entry into a new children array and then convert the
               current node to an internal node. */
            uint64_t other_hash = dictHashKey(d, node->entry.key);
            int other_bitindex = bitindex_at_level(other_hash, level);
            union dictNode *children = zmalloc(sizeof(union dictNode) * 1);
            children[0] = *node;
            node->sub.children = mask_childptr(children);
            node->sub.bitmap = (1 << other_bitindex);
            /* TODO (optimization): If bitindex != other_bitindex, then add our
               entry here directly, either before or after the other entry. */
        }
    }
}

/* Returns a pointer to an entry or NULL if the key isn't found in the dict. */
dictEntry *dictFind(dict *d, const void *key) {
    if (d->size == 0) return NULL; /* dict is empty */
    uint64_t hash = dictHashKey(d, key);
    int level = 0;
    union dictNode *node = &d->root;
    for (;;) {
        if (is_childptr(node->sub.children)) {
            /* It's an internal node. */
            union dictNode *children = unmask_childptr(node->sub.children);
            uint32_t bitmap = node->sub.bitmap;
            /* Use 5 bits of the hash as an index into one of 32 possible
               children. The child exists if the bit at bitindex is set. */
            int bitindex = bitindex_at_level(hash, level);
            uint32_t shifted = bitmap >> bitindex;
            if (shifted & 1) {
                /* Child exists. The position of the child is the number of
                 * 1-bits to the left of this bit in the bitmap. */
                int childpos = dict_popcount(shifted >> 1);
                node = &children[childpos];
                if (level * 5 >= 64) {
                    /* We've used up all hash bits. Use 2ndary hash function. */
                    assert(false); /* TODO */
                }
                level++;
            } else {
                /* Child doesn't exist. */
                return NULL;
            }
        } else {
            /* It's a leaf, i.e. a dictEntry. */
            if (dictCompareKeys(d, key, node->entry.key)) {
                return &node->entry;
            }
            /* There's another entry at this position. */
            return NULL;
        }
    }
}

/* Add or Overwrite:
 * Add an element, discarding the old value if the key already exists.
 * Return 1 if the key was added from scratch, 0 if there was already an
 * element with such key and dictReplace() just performed a value update
 * operation. */
int dictReplace(dict *d, void *key, void *val)
{
    dictEntry *entry, *existing, auxentry;

    /* Try to add the element. If the key
     * does not exists dictAdd will succeed. */
    entry = dictAddRaw(d,key,&existing);
    if (entry) {
        dictSetVal(d, entry, val);
        return 1;
    }

    /* Set the new value and free the old one. Note that it is important
     * to do that in this order, as the value may just be exactly the same
     * as the previous one. In this context, think to reference counting,
     * you want to increment (set), and then decrement (free), and not the
     * reverse. */
    auxentry = *existing;
    dictSetVal(d, existing, val);
    dictFreeVal(d, &auxentry);
    return 0;
}

/* Add or Find:
 * dictAddOrFind() is simply a version of dictAddRaw() that always
 * returns the hash entry of the specified key, even if the key already
 * exists and can't be added (in that case the entry of the already
 * existing key is returned.)
 *
 * See dictAddRaw() for more information. */
dictEntry *dictAddOrFind(dict *d, void *key) {
    dictEntry *entry, *existing;
    entry = dictAddRaw(d,key,&existing);
    return entry ? entry : existing;
}

/* Removes an entry from a node. Returns 1 if the key was found and 0 otherwise.
 * The 'deleted' entry is filled with the key and value from the deleted entry.
 * This is a recursive helper for dictGenericDelete(). */
int dictDeleteFromNode(dict *d, union dictNode *node, int level, uint64_t hash,
                       const void *key, dictEntry *deleted) {
    assert(is_childptr(node->sub.children));
    union dictNode *children = unmask_childptr(node->sub.children);
    uint32_t bitmap = node->sub.bitmap;
    /* Use 5 bits of the hash as an index into one of 32 possible
       children. The child exists if the bit at bitindex is set. */
    int bitindex = bitindex_at_level(hash, level);
    uint32_t shifted = bitmap >> bitindex;
    if (!(shifted & 1))
        return 0; /* Child doesn't exist. */

    /* The child index is the number of 1-bits to the left of this bit. */
    int childindex = dict_popcount(shifted >> 1);
    union dictNode *child = &children[childindex];
    if (is_entry(child)) {
        if (!dictCompareKeys(d, key, child->entry.key))
            return 0;
        /* It's a match. Fill 'deleted' and remove child from node. */
        if (deleted != NULL)
            *deleted = child->entry;
        int numchildren = dict_popcount(node->sub.bitmap);
        int numafter = numchildren - childindex;
        memmove(&children[childindex], &children[childindex + 1],
                sizeof(union dictNode) * numafter);
        children = zrealloc(children, sizeof(union dictNode) * --numchildren);
        node->sub.children = mask_childptr(children);
        node->sub.bitmap &= ~(1 << bitindex);
        if (numchildren > 1)
            return 1; /* Fast path. No need to collapse children. */
    } else {
        /* It's a subnode. Delete it from the child node. */
        if ((level + 1) * 5 >= 64) {
            /* All hash bits used up. TODO: Use secondary hash function. */
            assert(false);
        }
        if (!dictDeleteFromNode(d, child, level + 1, hash, key, deleted))
            return 0; /* Not found in subtree. */
    }

    /* If we're here, it means we have removed an entry. If the node has now
     * only one child which is an entry, we need to collapse the child.
     *
     *                                ,--entry
     * Before delete:  node---child--<
     *                                `--entry
     *
     * After delete:   node---child---entry
     *
     * After collapse: node---entry
     */
    if (dict_popcount(node->sub.bitmap) == 1 && is_entry(&children[0])) {
        *node = children[0];
        zfree(children);
    }
    return 1;
}

/* Search and remove an element. Returns 1 if the key was found and 0 otherwise.
 * The 'deleted' entry is filled with the key and value from the deleted entry.
 *
 * This is an helper function for dictDelete() and dictUnlink(). Please check
 * the top comment of those functions. */
int dictGenericDelete(dict *d, const void *key, dictEntry *deleted) {
    if (d->size == 1) {
        assert(is_entry(&d->root));
        if (dictCompareKeys(d, key, d->root.entry.key)) {
            *deleted = d->root.entry;
            d->size = 0;
            return 1;
        }
        return 0;
    }
    uint64_t hash = dictHashKey(d, key);
    return dictDeleteFromNode(d, &d->root, 0, hash, key, deleted);
}

/* Remove an element, returning DICT_OK on success or DICT_ERR if the
 * element was not found. */
int dictDelete(dict *d, const void *key) {
    dictEntry entry;
    if (dictGenericDelete(d, key, &entry)) {
        dictFreeKey(d, &entry);
        dictFreeVal(d, &entry);
        return DICT_OK;
    }
    return DICT_ERR;
}

/* Remove an element from the table, but without actually releasing
 * the key, value and dictionary entry. The dictionary entry is returned
 * if the element was found (and unlinked from the table), and the user
 * should later call `dictFreeUnlinkedEntry()` with it in order to release it.
 * Otherwise if the key is not found, NULL is returned.
 *
 * This function is useful when we want to remove something from the hash
 * table but want to use its value before actually deleting the entry.
 * Without this function the pattern would require two lookups:
 *
 *  entry = dictFind(...);
 *  // Do something with entry
 *  dictDelete(dictionary,entry);
 *
 * Thanks to this function it is possible to avoid this, and use
 * instead:
 *
 * entry = dictUnlink(dictionary,entry);
 * // Do something with entry
 * dictFreeUnlinkedEntry(entry); // <- This does not need to lookup again.
 */
dictEntry *dictUnlink(dict *d, const void *key) {
    dictEntry entry;
    if (dictGenericDelete(d, key, &entry)) {
        /* TODO: Rename dictGenericDelete() to dictUnlink() and refactor calls
           to it to get rid of the silly dup_entry. */
        dictEntry *dup_entry = zmalloc(sizeof(dictEntry));
        memcpy(dup_entry, &entry, sizeof(dictEntry));
        return dup_entry;
    }
    return NULL;
}

/* You need to call this function to really free the entry after a call
 * to dictUnlink(). It's safe to call this function with 'he' = NULL. */
void dictFreeUnlinkedEntry(dict *d, dictEntry *he) {
    if (he == NULL) return;
    dictFreeKey(d, he);
    dictFreeVal(d, he);
    zfree(he);
}

/* Deletes the contents of a node. Returns counter after incrementing it with
 * the number of deleted elements. */
int dictClearNode(dict *d, union dictNode *node, long counter,
                  void(callback)(void *)) {
    if (is_entry(node)) {
        dictFreeKey(d, &node->entry);
        dictFreeVal(d, &node->entry);
        counter++;
        if (callback && (counter & 65535) == 0)
            callback(d->privdata);
    } else {
        int numchildren = dict_popcount(node->sub.bitmap);
        union dictNode *children = unmask_childptr(node->sub.children);
        for (int i = 0; i < numchildren; i++)
            counter = dictClearNode(d, &children[i], counter, callback);
        zfree(children);
    }
    return counter;
}

/* Clear & Release the hash table */
void dictRelease(dict *d)
{
    if (d->size > 0) dictClearNode(d, &d->root, 0, NULL);
    zfree(d);
}

void *dictFetchValue(dict *d, const void *key) {
    dictEntry *he;

    he = dictFind(d,key);
    return he ? dictGetVal(he) : NULL;
}

dictIterator *dictGetIterator(dict *d)
{
    dictIterator *iter = zmalloc(sizeof(*iter));
    dictInitIterator(iter, d, 0);
    return iter;
}

dictIterator *dictGetSafeIterator(dict *d) {
    return dictGetIterator(d);
}

dictEntry *dictNextInNode(dictIterator *iter, union dictNode *node, int level) {
    assert(is_childptr(node->sub.children));
    union dictNode *children = unmask_childptr(node->sub.children);
    int bitmap = node->sub.bitmap;
    int bitindex = bitindex_at_level(iter->cursor, level);
    while (bitindex < 32) {
        int shifted = bitmap >> bitindex;
        if (shifted & 1) {
            /* Child exists. */
            int childindex = dict_popcount(shifted >> 1);
            union dictNode *child = &children[childindex];
            if (is_entry(child)) {
                /* Set start position for next time. */
                setBitindexAtLevel(&iter->cursor, level, ++bitindex);
                return &child->entry;
            } else {
                /* Find next recurively. */
                assert(level < 13); /* FIXME */
                dictEntry *found = dictNextInNode(iter, child, level + 1);
                if (found) return found;
            }
        }
        /* No more entries within child. Skip to beginning of next child. */
        setBitindexAtLevel(&iter->cursor, level, ++bitindex);
    }
    /* No more entries within node. Clear this and all sublevel indices. */
    iter->cursor &= (1 << (5 * level)) - 1;
    return NULL;
}

/* Returns a pointer to the next entry. It's safe to add, delete and replace
 * elements in the dict while iterating. However, the entry pointer returned by
 * this function becomes invalid when adding or deleting any entries. */
dictEntry *dictNext(dictIterator *iter) {
    switch (iter->d->size) {
    case 0:
        iter->cursor = 0;
        return NULL;
    case 1:
        assert(is_entry(&iter->d->root));
        if (iter->cursor == 0) {
            iter->cursor++;
            return &iter->d->root.entry;
        }
        iter->cursor = 0;
        return NULL;
    default:
        return dictNextInNode(iter, &iter->d->root, 0);
    }
}

void dictInitIterator(dictIterator *iter, dict *d, uint64_t cursor) {
    iter->d = d;
    iter->cursor = cursor;
}

void dictReleaseIterator(dictIterator *iter)
{
    zfree(iter);
}

/* Return a random entry from the hash table. Useful to
 * implement randomized algorithms */
dictEntry *dictGetRandomKey(dict *d)
{
    if (dictSize(d) == 0) return NULL;
    dictIterator iter;
    uint64_t start = randomULong();
    dictInitIterator(&iter, d, start);
    dictEntry *entry = dictNext(&iter);
    if (entry == NULL) {
        /* warp and start from the beginning */
        assert(iter.cursor == 0);
        entry = dictNext(&iter);
        assert(entry != NULL);
    }
    return entry;
}

/* This function samples the dictionary to return a few keys from random
 * locations.
 *
 * It does not guarantee to return all the keys specified in 'count', nor
 * it does guarantee to return non-duplicated elements, however it will make
 * some effort to do both things.
 *
 * Returned pointers to hash table entries are stored into 'des' that
 * points to an array of dictEntry pointers. The array must have room for
 * at least 'count' elements, that is the argument we pass to the function
 * to tell how many random elements we need.
 *
 * The function returns the number of items stored into 'des', that may
 * be less than 'count' if the hash table has less than 'count' elements
 * inside, or if not enough elements were found in a reasonable amount of
 * steps.
 *
 * Note that this function is not suitable when you need a good distribution
 * of the returned items, but only when you need to "sample" a given number
 * of continuous elements to run some kind of algorithm or to produce
 * statistics. However the function is much faster than dictGetRandomKey()
 * at producing N elements. */
unsigned int dictGetSomeKeys(dict *d, dictEntry **des, unsigned int count) {
    if (dictSize(d) < count) count = dictSize(d);
    dictIterator iter;
    uint64_t start = randomULong();
    dictInitIterator(&iter, d, start);
    for (unsigned int i = 0; i < count; i++) {
        dictEntry *entry = dictNext(&iter);
        if (entry == NULL) {
            /* warp and start from the beginning */
            assert(iter.cursor == 0);
            entry = dictNext(&iter);
            assert(entry != NULL);
        }
        des[i] = entry;
    }
    return count;
}

/* This is like dictGetRandomKey() from the POV of the API, but will do more
 * work to ensure a better distribution of the returned element.
 *
 * This function improves the distribution because the dictGetRandomKey()
 * problem is that it selects a random bucket, then it selects a random
 * element from the chain in the bucket. However elements being in different
 * chain lengths will have different probabilities of being reported. With
 * this function instead what we do is to consider a "linear" range of the table
 * that may be constituted of N buckets with chains of different lengths
 * appearing one after the other. Then we report a random element in the range.
 * In this way we smooth away the problem of different chain lengths. */
#define GETFAIR_NUM_ENTRIES 15
dictEntry *dictGetFairRandomKey(dict *d) {
    dictEntry *entries[GETFAIR_NUM_ENTRIES];
    unsigned int count = dictGetSomeKeys(d,entries,GETFAIR_NUM_ENTRIES);
    /* Note that dictGetSomeKeys() may return zero elements in an unlucky
     * run() even if there are actually elements inside the hash table. So
     * when we get zero, we call the true dictGetRandomKey() that will always
     * yield the element if the hash table has at least one. */
    if (count == 0) return dictGetRandomKey(d);
    unsigned int idx = rand() % count;
    return entries[idx];
}

/* dictScan() is used to iterate over the elements of a dictionary.
 *
 * Iterating works the following way:
 *
 * 1) Initially you call the function using a cursor (v) value of 0.
 * 2) The function performs one step of the iteration, and returns the
 *    new cursor value you must use in the next call.
 * 3) When the returned cursor is 0, the iteration is complete.
 *
 * The function guarantees all elements present in the
 * dictionary get returned between the start and end of the iteration.
 * However it is possible some elements get returned multiple times.
 *
 * For every element returned, the callback argument 'fn' is
 * called with 'privdata' as first argument and the dictionary entry
 * 'de' as second argument.
 *
 * HOW IT WORKS.
 *
 * The cursor is basically a path into the hash tree. The keys are iterated in
 * the order of how they are stored in the hash tree, in depth first order. The
 * 5 least significant bits of the cursor are used as an index into the first
 * level of nodes. The next 5 bits are used an index into the next level and so
 * forth.
 *
 * LIMITATIONS
 *
 * This iterator is completely stateless, and this is a huge advantage,
 * including no additional memory used.
 *
 * It also has the following nice properties:
 *
 * 1) No duplicates. Each element is returned only once.
 *
 * 2) The iterator can usually return exactly the requested number of entries.
 *    The exception is when there are multiple keys with exactly the same 64-bit
 *    hash value. These are always returned together, since they correspond to
 *    the same cursor value.
 *
 * The disadvantages resulting from this design is:
 *
 * 1) The cursor is somewhat hard to understand at first, but this comment is
 *    supposed to help.
 */
unsigned long dictScan(dict *d,
                       unsigned long v,
                       dictScanFunction *fn,
                       dictScanBucketFunction* bucketfn,
                       void *privdata)
{
    DICT_NOTUSED(bucketfn);
    dictIterator iter;
    dictInitIterator(&iter, d, v);
    unsigned long u = v;
    do {
        const dictEntry *de = dictNext(&iter);
        if (de == NULL)
            return 0;
        fn(privdata, de);
        u = iter.cursor;
    } while (u == v);
    return u;
}

void dictEmpty(dict *d, void(callback)(void*)) {
    if (d->size > 0) dictClearNode(d, &d->root, 0, callback);
    d->size = 0;
}

void dictEnableResize(void) {
    dict_can_resize = 1;
}

void dictDisableResize(void) {
    dict_can_resize = 0;
}

int dictExpand(dict *d, unsigned long size) {
    DICT_NOTUSED(d);
    DICT_NOTUSED(size);
    return DICT_OK;
}
int dictTryExpand(dict *d, unsigned long size) {
    DICT_NOTUSED(d);
    DICT_NOTUSED(size);
    return DICT_OK;
}
int dictResize(dict *d) {
    DICT_NOTUSED(d);
    return DICT_OK;
}

uint64_t dictGetHash(dict *d, const void *key) {
    return dictHashKey(d, key);
}

/* Finds the dictEntry reference by using pointer and pre-calculated hash.
 * oldkey is a dead pointer and should not be accessed.
 * the hash value should be provided using dictGetHash.
 * no string / key comparison is performed.
 * return value is the reference to the dictEntry if found, or NULL if not found. */
dictEntry **dictFindEntryRefByPtrAndHash(dict *d, const void *oldptr, uint64_t hash) {
    DICT_NOTUSED(oldptr);
    DICT_NOTUSED(hash);
    if (dictSize(d) == 0) return NULL; /* dict is empty */
    /* FIXME */
    return NULL;
}

/* Returns the estimated memory usage of the dict structure in bytes. This does
 * NOT include additional memory allocated for keys and values.
 *
 * With large sizes, the root node will be close to full. So will the nodes
 * close to the root. As a rough estimate we assume that half of the nodes, the
 * ones far away from the root, are only half-full, thus roughly 1.5 * log32 N.
 *
 * (The estimate 1.28 * N given in Phil Bagwell (2000). Ideal Hash Trees,
 * Section 3.5 "Space Used" refers to a HAMT with resizable root table.)
 */
size_t dictEstimateMem(dict *d) {
    /* Log base conversion rule: log32(x) = log2(x) / log2(32) */
    unsigned long numnodes = 1.5 * log2(d->size) / log2(32);
    return sizeof(dict) + sizeof(union dictNode) * numnodes;
}

/* ------------------------------- Debugging ---------------------------------*/

/* #define DICT_STATS_VECTLEN 50 */
/* size_t _dictGetStatsHt(char *buf, size_t bufsize, dictht *ht, int tableid) { */
/*     unsigned long i, slots = 0, chainlen, maxchainlen = 0; */
/*     unsigned long totchainlen = 0; */
/*     unsigned long clvector[DICT_STATS_VECTLEN]; */
/*     size_t l = 0; */

/*     if (ht->used == 0) { */
/*         return snprintf(buf,bufsize, */
/*             "No stats available for empty dictionaries\n"); */
/*     } */

/*     /\* Compute stats. *\/ */
/*     for (i = 0; i < DICT_STATS_VECTLEN; i++) clvector[i] = 0; */
/*     for (i = 0; i < ht->size; i++) { */
/*         dictEntry *he; */

/*         if (ht->table[i] == NULL) { */
/*             clvector[0]++; */
/*             continue; */
/*         } */
/*         slots++; */
/*         /\* For each hash entry on this slot... *\/ */
/*         chainlen = 0; */
/*         he = ht->table[i]; */
/*         while(he) { */
/*             chainlen++; */
/*             he = he->next; */
/*         } */
/*         clvector[(chainlen < DICT_STATS_VECTLEN) ? chainlen : (DICT_STATS_VECTLEN-1)]++; */
/*         if (chainlen > maxchainlen) maxchainlen = chainlen; */
/*         totchainlen += chainlen; */
/*     } */

/*     /\* Generate human readable stats. *\/ */
/*     l += snprintf(buf+l,bufsize-l, */
/*         "Hash table %d stats (%s):\n" */
/*         " table size: %lu\n" */
/*         " number of elements: %lu\n" */
/*         " different slots: %lu\n" */
/*         " max chain length: %lu\n" */
/*         " avg chain length (counted): %.02f\n" */
/*         " avg chain length (computed): %.02f\n" */
/*         " Chain length distribution:\n", */
/*         tableid, (tableid == 0) ? "main hash table" : "rehashing target", */
/*         ht->size, ht->used, slots, maxchainlen, */
/*         (float)totchainlen/slots, (float)ht->used/slots); */

/*     for (i = 0; i < DICT_STATS_VECTLEN-1; i++) { */
/*         if (clvector[i] == 0) continue; */
/*         if (l >= bufsize) break; */
/*         l += snprintf(buf+l,bufsize-l, */
/*             "   %s%ld: %ld (%.02f%%)\n", */
/*             (i == DICT_STATS_VECTLEN-1)?">= ":"", */
/*             i, clvector[i], ((float)clvector[i]/ht->size)*100); */
/*     } */

/*     /\* Unlike snprintf(), return the number of characters actually written. *\/ */
/*     if (bufsize) buf[bufsize-1] = '\0'; */
/*     return strlen(buf); */
/* } */

void dictGetStats(char *buf, size_t bufsize, dict *d) {
    DICT_NOTUSED(d);
    snprintf(buf, bufsize, "Stats for HAMT not implemented");
    /* Make sure there is a NULL term at the end. */
    if (bufsize) buf[bufsize-1] = '\0';
}

/* ------------------------------- Benchmark ---------------------------------*/

#ifdef REDIS_TEST

uint64_t hashCallback(const void *key) {
    return dictGenHashFunction((unsigned char*)key, strlen((char*)key));
}

int compareCallback(void *privdata, const void *key1, const void *key2) {
    int l1,l2;
    DICT_NOTUSED(privdata);

    l1 = strlen((char*)key1);
    l2 = strlen((char*)key2);
    if (l1 != l2) return 0;
    return memcmp(key1, key2, l1) == 0;
}

void freeCallback(void *privdata, void *val) {
    DICT_NOTUSED(privdata);

    zfree(val);
}

char *stringFromLongLong(long long value) {
    char buf[32];
    int len;
    char *s;

    len = sprintf(buf,"%lld",value);
    s = zmalloc(len+1);
    memcpy(s, buf, len);
    s[len] = '\0';
    return s;
}

dictType BenchmarkDictType = {
    hashCallback,
    NULL,
    NULL,
    compareCallback,
    freeCallback,
    NULL,
    NULL
};

#define start_benchmark() start = timeInMilliseconds()
#define end_benchmark(msg) do { \
    elapsed = timeInMilliseconds()-start; \
    printf(msg ": %ld items in %lld ms\n", count, elapsed); \
} while(0)

/* ./redis-server test dict [<count> | --accurate] */
int dictTest(int argc, char **argv, int accurate) {
    long j;
    long long start, elapsed;
    dict *dict = dictCreate(&BenchmarkDictType,NULL);
    long count = 0;

    if (argc == 4) {
        if (accurate) {
            count = 5000000;
        } else {
            count = strtol(argv[3],NULL,10);
        }
    } else {
        count = 5000;
    }

    start_benchmark();
    for (j = 0; j < count; j++) {
        int retval = dictAdd(dict,stringFromLongLong(j),(void*)j);
        assert(retval == DICT_OK);
    }
    end_benchmark("Inserting");
    assert((long)dictSize(dict) == count);

    /* Wait for rehashing. */
    while (dictIsRehashing(dict)) {
        dictRehashMilliseconds(dict,100);
    }

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(j);
        dictEntry *de = dictFind(dict,key);
        assert(de != NULL);
        zfree(key);
    }
    end_benchmark("Linear access of existing elements");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(j);
        dictEntry *de = dictFind(dict,key);
        assert(de != NULL);
        zfree(key);
    }
    end_benchmark("Linear access of existing elements (2nd round)");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(rand() % count);
        dictEntry *de = dictFind(dict,key);
        assert(de != NULL);
        zfree(key);
    }
    end_benchmark("Random access of existing elements");

    start_benchmark();
    for (j = 0; j < count; j++) {
        dictEntry *de = dictGetRandomKey(dict);
        assert(de != NULL);
    }
    end_benchmark("Accessing random keys");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(rand() % count);
        key[0] = 'X';
        dictEntry *de = dictFind(dict,key);
        assert(de == NULL);
        zfree(key);
    }
    end_benchmark("Accessing missing");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(j);
        int retval = dictDelete(dict,key);
        assert(retval == DICT_OK);
        key[0] += 17; /* Change first number to letter. */
        retval = dictAdd(dict,key,(void*)j);
        assert(retval == DICT_OK);
    }
    end_benchmark("Removing and adding");
    dictRelease(dict);
    return 0;
}
#endif
