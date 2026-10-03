// solver for Zachtronics Cribbage Solitaire:
// https://zachtronics.com/solitaire-rules/#logo-cheyenne

use std::mem::swap;
use std::range::Range;

#[inline(always)]
fn nibble_at(within: impl Into<u64>, n: impl Into<u32>) -> u8 {
    ((within.into() >> n.into() * 4) & 0xF) as u8
}

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
struct Stack(u64);

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

    fn score_run(self) -> u8 {
        let mask_at = |n| 1u16 << nibble_at(self.0, n);
        let mut best = 0;
        let mut mask = mask_at(0) | mask_at(1);
        for n in 2..=6 {
            mask |= mask_at(n);
            // remove the bit representing 0
            let a = mask >> 1;
            // isolate the least-significant set bit: there definitely is one
            // because this is never called on an empty stack
            let min_bit = a.isolate_lowest_one();
            // collapse the least-significant run upward, leaving the next higher
            // bit set; can't overflow because there's always at least one clear
            // bit due to the above shift
            let b = a + min_bit;
            // if there was exactly one run of n bits set, then that run is now
            // collapsed to a single bit n bits above min_bit
            let n = n + 1;
            if min_bit << n == b {
                best = n;
            }
        }
        best
    }
}

struct InitialTableaux([u64; 4]);

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
struct Tableaux(u16);

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

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
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

    fn play_card(mut self, card: u8) -> Option<Self> {
        self.total += card.min(10);
        if self.total > 31 {
            return None;
        }
        self.stack.push(card);
        self.score += self.score_total();
        self.score += self.stack.score_first_jack();
        self.score += self.stack.score_repeats();
        self.score += self.stack.score_run();
        Some(self)
    }

    fn move_from(mut self, initial: &InitialTableaux, column: u8) -> Option<Self> {
        self.tableaux.0 = self.tableaux.0.wrapping_sub(1 << column * 4);
        let offset = nibble_at(self.tableaux.0, column);
        if offset == 0xF {
            // underflow, no cards left in this column
            None
        } else {
            let card = nibble_at(initial.0[usize::from(column)], offset);
            self.play_card(card)
        }
    }
}

impl std::fmt::Debug for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Position {{ tableaux: {:x}, stack: {:x}, total: {}, score: {} }}",
            self.tableaux.0, self.stack.0, self.total, self.score
        )
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
}

impl Default for StackCache {
    fn default() -> Self {
        Self {
            todo: Vec::new(),
            next: Vec::new(),
            done: Vec::new(),
            cache_ranges: Box::new(std::array::repeat((0..0).into())),
            cache: Vec::new(),
        }
    }
}

impl StackCache {
    fn uncached(&mut self, initial: &InitialTableaux, tableaux: Tableaux) {
        self.todo.push(Position {
            tableaux,
            stack: Stack::default(),
            total: 0,
            score: 0,
        });
        while !self.todo.is_empty() {
            for position in self.todo.drain(..) {
                let past = self.next.len();
                self.next
                    .extend((0..4).filter_map(|column| position.move_from(initial, column)));
                if past == self.next.len() {
                    match self.done.last_mut() {
                        Some(prev) if position.tableaux == prev.tableaux => {
                            if position.score > prev.score {
                                *prev = position;
                            }
                        }
                        _ => self.done.push(position),
                    }
                }
            }
            swap(&mut self.todo, &mut self.next);
            self.todo.sort_unstable();
            self.todo.dedup();
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
        }
        eprintln!(
            "searched {leaves} solutions, {} edges {edges} times ({}x)",
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
