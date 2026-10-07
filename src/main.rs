// solver for Zachtronics Cribbage Solitaire:
// https://zachtronics.com/solitaire-rules/#logo-cheyenne

use std::array;
use std::range::Range;

use fearless_simd::prelude::*;
use fearless_simd::{Level, Simd, dispatch, i16x8, u8x16, u16x8, u64x2};
use fearless_simd_macros::simd;

#[inline(always)]
fn nibble_at(within: impl Into<u64>, n: impl Into<u32>) -> u8 {
    ((within.into() >> n.into() * 4) & 0xF) as u8
}

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
struct Stack(u64);

impl std::fmt::Debug for Stack {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:x}", self.0)
    }
}

impl Stack {
    fn push(&mut self, card: u8) {
        self.0 = self.0 << 4 | card as u64;
    }

    fn score_first_jack(self) -> u8 {
        if self.0 == 11 { 2 } else { 0 }
    }

    fn score_repeats(self) -> u8 {
        let repeated = (self.0 & 0xF) * 0x11111;
        let matched = self.0 ^ repeated;
        const SCORES: u32 = 2 << (2 * 4) | 6 << (3 * 4) | 12 << (4 * 4);
        nibble_at(SCORES, matched.trailing_zeros() / 4)
    }

    #[simd]
    fn score_run_simd<S: Simd>(self, simd: S) -> u8 {
        // turn card ranks into bitmasks, where rank N sets bit N-1, and an
        // absent card (rank 0) doesn't set any bits. conceptually, this is a
        // left-shift of a constant by a variable number of bits, followed by a
        // right-shift by 1 bit.
        //
        // LLVM and fearless-simd pull off an impressive variety of tricks if we
        // ask for that: AVX512 has an instruction for it; on SSE4.2 it adds the
        // shift amount to the exponent field in 1.0xf32 and converts that to
        // an int; AVX2 did some magic I didn't understand. but there are only
        // 16 possible outputs, which is easy to turn into a small lookup table,
        // and those three instruction sets also have reasonable support for
        // swizzles which can be used as table lookup.
        //
        // however, baseline x86-64 (SSE) doesn't do either swizzles or variable
        // shifts. if we put the stack into a vector but then fall back to
        // per-lane scalar operations, the resulting code is, uh, bad. so just
        // do everything preceding the variable shift as scalar ops to begin
        // with.
        //
        // while the card ranks are packed into 4 bits each, the corresponding
        // bitmask needs to be 16 bits wide. the most portable vector length is
        // 128 bits, which only has room for 8 bitmasks. fortunately, we only
        // need to score the most recent 7 cards.
        let masks = match simd.level() {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            level if level.as_sse4_2().is_some() => {
                // zero-extend packed 4-bit values to 8-bit
                let nibbles = u64x2::simd_from(simd, [self.0, 0]).to_bytes();
                let bytes = nibbles.zip_low(nibbles >> 4) & 0xF;
                // create lookup tables of shift results
                let shifts = array::from_fn(|i| (1u16 << i) >> 1);
                let shifts_lo = shifts.map(|n| n as u8);
                let shifts_hi = shifts.map(|n| (n >> 8) as u8);
                // look up each card rank in the tables
                let bytes_lo = u8x16::simd_from(simd, shifts_lo).swizzle_dyn(bytes);
                let bytes_hi = u8x16::simd_from(simd, shifts_hi).swizzle_dyn(bytes);
                u16x8::from_bytes(bytes_lo.zip_low(bytes_hi))
            }
            // on other targets, do the shifts in scalar registers first
            _ => u16x8::from_fn(simd, |i| 1 << nibble_at(self.0, i as u8)) >> 1,
        };

        // accumulate masks so each has all the bits from newer cards
        let masks = masks | masks.shift_elements_right::<1>(0);
        let masks = masks | masks.shift_elements_right::<2>(0);
        let masks = masks | masks.shift_elements_right::<4>(0);
        // isolate the least-significant set bit: there definitely is one
        // because this is never called on an empty stack
        let masks = masks.bitcast::<i16x8<S>>();
        let min_bits = masks & -masks;
        // collapse the least-significant run upward, leaving the next higher
        // bit set; can't overflow because there's always at least one clear bit
        let collapsed = masks + min_bits;
        // if there was exactly one run of n bits set, then that run is now
        // collapsed to a single bit n bits above min_bit. we only want to
        // score runs of length 3 through 7, so don't shift the other lanes:
        // that guarantees they can't match with the mask which had the min-bit
        // set. (this is conceptually another variable left-shift, but because
        // the shift amount in each lane is constant, we can transform it to
        // multiplication by a constant vector instead. on AVX512, LLVM actually
        // transforms this back to a variable shift, but multiplication is
        // widely available and a safe bet otherwise.)
        let expected = min_bits * i16x8::simd_from(simd, [0, 0, 3, 4, 5, 6, 7, 0].map(|i| 1 << i));
        let matched = collapsed.simd_eq(expected).to_bitmask();
        // score based on the highest-numbered lane which matched, if any
        matched.highest_one().map_or(0, |x| (x + 1) as _)
    }
}

struct InitialTableaux([u64; 4]);

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
struct Tableaux(u16);

impl std::fmt::Debug for Tableaux {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:04x}", self.0)
    }
}

impl Tableaux {
    fn new(initial: &InitialTableaux) -> Tableaux {
        let mut tableaux = 0;
        for (column, contents) in initial.0.iter().enumerate() {
            let height = (64 - contents.leading_zeros() as u16) / 4;
            tableaux |= height << column * 4;
        }
        Tableaux(tableaux)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Position {
    tableaux: Tableaux,
    stack: Stack,
    total: u8,
    score: u8,
}

impl Position {
    fn score_total(&self) -> u8 {
        if self.total & 0xF == 0xF { 2 } else { 0 }
    }

    #[inline]
    fn update_score(&mut self, level: Level) {
        self.score += self.score_total();
        self.score += self.stack.score_first_jack();
        self.score += self.stack.score_repeats();
        self.score += dispatch!(level, simd => self.stack.score_run_simd(simd));
    }

    fn move_from(mut self, initial: &InitialTableaux, column: u8) -> Option<Self> {
        self.tableaux.0 = self.tableaux.0.wrapping_sub(1 << column * 4);
        let offset = nibble_at(self.tableaux.0, column);
        if offset == 0xF {
            // underflow, no cards left in this column
            return None;
        }
        let card = nibble_at(initial.0[usize::from(column)], offset);
        self.total += card.min(10);
        if self.total > 31 {
            return None;
        }
        self.stack.push(card);
        Some(self)
    }
}

#[derive(Clone, Copy, Default)]
struct BestStack {
    score: u16,
    tableaux: Tableaux,
}

struct StackCache {
    todo: Vec<Position>,
    next: Vec<Position>,
    done: Vec<Position>,
    cache_ranges: Box<[Range<u32>; 65536]>,
    cache: Vec<BestStack>,
    unique: u16,
}

impl Default for StackCache {
    fn default() -> Self {
        Self {
            todo: Vec::new(),
            next: Vec::new(),
            done: Vec::new(),
            cache_ranges: Box::new(array::repeat((0..0).into())),
            cache: Vec::new(),
            unique: 0,
        }
    }
}

fn merge_many_dedup(
    iter: impl Iterator<Item = impl Iterator<Item = Position>>,
) -> impl Iterator<Item = Position> {
    let mut nexts = Vec::from_iter(iter.filter_map(|mut iter| Some((iter.next()?, iter))));
    std::iter::from_fn(move || {
        let (next_idx, _) = nexts
            .iter()
            .enumerate()
            .min_by(|(_, (a, _)), (_, (b, _))| {
                a.tableaux.cmp(&b.tableaux).then(a.stack.cmp(&b.stack))
            })?;
        let mut next = nexts.swap_remove(next_idx);
        nexts.retain_mut(|other| {
            if next.0.tableaux != other.0.tableaux || next.0.stack != other.0.stack {
                true
            } else if let Some(future) = other.1.next() {
                other.0 = future;
                true
            } else {
                false
            }
        });
        if let Some(future) = next.1.next() {
            nexts.push((future, next.1));
        }
        Some(next.0)
    })
}

fn best_dedup(mut iter: impl Iterator<Item = Position>) -> impl Iterator<Item = Position> {
    let mut next = iter.next();
    std::iter::from_fn(move || {
        let mut a = next.take()?;
        while let Some(b) = iter.next() {
            if a.tableaux != b.tableaux {
                next = Some(b);
                break;
            }
            if a.score < b.score {
                a = b;
            }
        }
        Some(a)
    })
}

impl StackCache {
    fn uncached(&mut self, initial: &InitialTableaux, tableaux: Tableaux) {
        let level = Level::new();
        self.todo.push(Position {
            tableaux,
            ..Position::default()
        });
        while !self.todo.is_empty() {
            // we're keeping self.todo sorted and unique by tableaux then stack.
            // move_from preserves that sort order and uniqueness if given
            // the same column number every time: it subtracts a constant from
            // tableaux, and inserts a nibble at the bottom of stack. so if we
            // generate moves for each column separately, we have four sorted
            // lists, which can be merged and deduplicated in linear time.
            self.next.extend(
                merge_many_dedup((0..4).map(|column| {
                    self.todo
                        .iter()
                        .filter_map(move |position| position.move_from(initial, column))
                }))
                .map(|mut position| {
                    position.update_score(level);
                    position
                }),
            );

            self.done
                .extend(best_dedup(self.todo.drain(..).filter(|position| {
                    (0..4).all(|column| position.move_from(initial, column).is_none())
                })));

            std::mem::swap(&mut self.todo, &mut self.next);
        }

        // try the highest scores first, but in case of ties, try the smallest
        // stacks first, with the idea that playing fewer cards leaves more
        // opportunities for high scores later
        self.done
            .sort_unstable_by(|x, y| x.score.cmp(&y.score).then(x.stack.cmp(&y.stack).reverse()));
    }

    fn cached(&mut self, initial: &InitialTableaux, tableaux: Tableaux) -> &[BestStack] {
        let mut range = self.cache_ranges[usize::from(tableaux.0)];
        if range.start == range.end {
            self.uncached(initial, tableaux);
            self.unique += 1;
            //eprintln!("({:x},{:?})", tableaux.0, self.done);

            range.start = self.cache.len().try_into().unwrap();
            self.cache
                .extend(self.done.drain(..).map(|position| BestStack {
                    score: position.score.into(),
                    tableaux: position.tableaux,
                }));
            range.end = self.cache.len().try_into().unwrap();
            if range.start == range.end {
                range.start += 1;
            }
            self.cache_ranges[usize::from(tableaux.0)] = range;
        }

        &self.cache[range.start.min(range.end) as usize..range.end as usize]
    }
}

struct SearchState {
    stacks: StackCache,
    best: Box<[BestStack; 65536]>,
    initial: InitialTableaux,
}

impl SearchState {
    fn new(initial: InitialTableaux) -> SearchState {
        let mut stacks = StackCache::default();
        let mut best = Box::new([BestStack::default(); 65536]);
        let start = Tableaux::new(&initial);
        best[usize::from(start.0)] = BestStack {
            score: 1,
            tableaux: start,
        };
        // once the piles are empty, there are no moves to make
        stacks.cache_ranges[0].start = 1;
        SearchState {
            stacks,
            best,
            initial,
        }
    }

    fn search(&mut self) -> u16 {
        let mut edges = 0;
        let mut leaves = 0;
        let mut todo = vec![Tableaux::new(&self.initial)];
        let mut max_todo = todo.len();
        while let Some(cur) = todo.pop() {
            let cur_best = self.best[usize::from(cur.0)].score;
            for next in self.stacks.cached(&self.initial, cur) {
                if next.tableaux == Tableaux::default() {
                    leaves += 1;
                }
                let new_score = cur_best + next.score;
                if new_score > self.best[usize::from(next.tableaux.0)].score {
                    if next.tableaux == Tableaux::default() {
                        eprintln!(" *** HIGH SCORE {}", new_score - 1);
                    }
                    self.best[usize::from(next.tableaux.0)] = BestStack {
                        score: new_score,
                        tableaux: cur,
                    };
                    todo.push(next.tableaux);
                    edges += 1;
                }
            }
            max_todo = todo.len().max(max_todo);
        }
        eprintln!(
            "allocated capacity: todo {}/{}, stacks.todo {}, stacks.next {}, stacks.done {}",
            max_todo,
            todo.capacity(),
            self.stacks.todo.capacity(),
            self.stacks.next.capacity(),
            self.stacks.done.capacity()
        );
        eprintln!(
            "searched {leaves} solutions, {} nodes, {} edges {edges} times ({}x)",
            self.stacks.unique,
            self.stacks.cache.len(),
            edges as f64 / self.stacks.cache.len() as f64
        );
        self.best[0].score - 1
    }
}

fn main() {
    let initial = InitialTableaux([0xdcba987654321; 4]);
    let score = SearchState::new(initial).search();
    println!("{score}");
}
