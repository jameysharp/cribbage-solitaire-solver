{-# OPTIONS_GHC -funbox-strict-fields -Wall #-}
-- useful build flags: -O3 -rtsopts -finfo-table-map -ddump-to-file -ddump-simpl -ddump-stg-final -ddump-asm
-- useful RTS flags: -s, or -l with -hi or -hT

-- Zachtronics Cribbage Solitaire
-- https://zachtronics.com/solitaire-rules/#logo-cheyenne

import Control.Exception (assert)
import Control.Monad
import Control.Monad.ST
import Data.Array.IArray
import Data.Array.ST
import Data.Array.Unboxed
import Data.Bits
import Data.Char
import Data.List
import Data.Maybe
import Data.Ord
import qualified Data.Set as Set
import Data.Word
import Debug.Trace
import Numeric

newtype Card = Card { rank :: Word8 } deriving (Eq, Ord)

instance Show Card where
    showsPrec _ c s = ("A23456789TJQK" !! fromIntegral (rank c - 1)) : s

fromRank :: (Integral a, Show a) => a -> Card
fromRank x = assert (x >= 1 && x <= 13) (Card (fromIntegral x))

fromLabel :: Char -> Card
fromLabel c = case toUpper c of
    'A' -> Card 1
    'T' -> Card 10
    'J' -> Card 11
    'Q' -> Card 12
    'K' -> Card 13
    _ -> fromRank (digitToInt c)

value :: Card -> Word8
-- value c = min 10 (rank c)
value c = fromIntegral z where
    x = fromIntegral (rank c) :: Int
    y = 10 - x
    isNeg = y `shiftR` (finiteBitSize y - 1)
    z = x + isNeg .&. y

newtype PackedStack = PackedStack Word64 deriving (Eq, Ord)

instance Show PackedStack where
    showsPrec _ (PackedStack p) = showHex p

pushStack :: PackedStack -> Card -> PackedStack
pushStack (PackedStack s) c = PackedStack (s `shiftL` 4 .|. fromIntegral (rank c))

countRepeated :: PackedStack -> Word8
countRepeated (PackedStack s) = (0xc6200 :: Int) `nibbleAt` (countTrailingZeros matched `div` 4) where
    repeated = (s .&. 0xF) * 0x11111
    matched = s `xor` repeated

runLength :: PackedStack -> Int
runLength (PackedStack s) = fst $ foldl' go (0, maskAt 0 .|. maskAt 1) [2..6] where
    -- safety: nibbleAt returns 0..15 so this shift is in-bounds for any type
    -- at least 16 bits wide. choose the machine Int type because otherwise GHC
    -- does unnecessary masking or overflow checks
    maskAt n = 1 `unsafeShiftL` fromIntegral (s `nibbleAt` n) :: Int
    go (!best, !prev) n = (best', next) where
        n' = n + 1
        next = prev .|. maskAt n
        -- remove the bit representing 0
        a = next `shiftR` 1
        -- isolate the least-significant set bit: there definitely is one
        -- because this is never called on an empty stack
        minBit = a .&. negate a
        -- collapse the least-significant run upward, leaving the next higher
        -- bit set; can't overflow because there's always at least one clear
        -- bit due to the above shift
        b = a + minBit
        -- if there was exactly one run of n' bits set, then that run is now
        -- collapsed to a single bit n' bits above minBit. safety: n' <= 7
        -- isRun = minBit `unsafeShiftL` n' == b
        mask = nonZero ((minBit `unsafeShiftL` n') `xor` b)
        -- best' = if isRun then n' else best
        best' = (complement mask .&. n') .|. (mask .&. best)

nibbleAt :: (Bits a, Integral a) => a -> Int -> Word8
bits `nibbleAt` n = fromIntegral ((bits `unsafeShiftR` (n * 4)) .&. 0xF)

type Score = Word8
data Stack = Stack
    { stack :: !PackedStack
    , stackTotal :: !Word8
    , score :: !Score
    } deriving (Show)

emptyStack :: Stack
emptyStack = Stack { stack = PackedStack 0, stackTotal = 0, score = 0 }

play :: Stack -> Card -> Maybe Stack
play p c = if stackTotal' > 31 then Nothing else Just $ Stack { stack = stack', stackTotal = stackTotal', score = score' } where
    stack' = pushStack (stack p) c
    stackTotal' = value c + stackTotal p
    -- scoring rules:
    score' = fromIntegral $ firstJack + stackTotal15 + sameCard + runPoints + fromIntegral (score p)
    -- 2 points if first card played on stack is a Jack, branchless version
    firstJack = (nonZero zeroIfScore `shiftL` 1) + 2 where
        zeroIfEmpty = fromIntegral (stackTotal p)
        zeroIfJack = fromIntegral (rank c) `xor` 11
        zeroIfScore = zeroIfEmpty .|. zeroIfJack
    -- 2 points if stack total is 15 or 31, branchless version
    stackTotal15 = z where
        x = fromIntegral stackTotal' .&. 0xF :: Int
        y = x .&. (x `shiftR` 2)
        z = y .&. (y `shiftL` 1)
    -- most recent cards had the same rank
    sameCard = fromIntegral (countRepeated stack')
    -- consecutive run of cards, played in any order
    runPoints = runLength stack'

-- branchless helper that returns 0 if x is 0, and -1 otherwise
nonZero :: Int -> Int
nonZero x = (x .|. negate x) `shiftR` (finiteBitSize x - 1)

playTests :: [Bool]
playTests =
    [ stackTotal (plays [c]) == value (fromLabel c) | c <- "A23456789TJQK" ] ++
    [ score (plays cs) == 2 | cs <- ["J", "78", "TQAT"] ] ++
    [ score (plays [c, c]) == 2 | c <- "A23456789T" ++ "QK" ] ++
    [ score (plays [c, c, c]) == 8 | c <- "A234" ++ "6789T" ++ "QK" ] ++
    [ score (plays ['A', c, c, c]) == 10 | c <- "TJQK" ] ++
    [ score (plays [c, c, c, 'A']) == 10 | c <- "5TQK" ] ++
    [ score (plays ['2', c, c, c, c]) == 20 | c <- "A34567" ] ++
    [ score (plays cs) == 3 | cs <- permutations "567" ] ++
    [ score (plays (cs ++ "6")) == 7 | cs <- permutations "345" ] ++
    [ score (plays (cs ++ "56")) == 12 | cs <- permutations "234" ] ++
    [ score (plays (cs ++ "456")) == 20 | cs <- permutations "A23" ]

{-# NOINLINE plays #-}
plays :: [Char] -> Stack
plays labels = case foldM play emptyStack $ map fromLabel labels of
    Nothing -> error ("can't play " ++ labels)
    Just result -> result

newtype InitialTableaux = InitialTableaux (UArray Int Word64)

instance Show InitialTableaux where
    show (InitialTableaux t) = 'x' : foldrArray showPile "" t where
        showPile s "" = showHex s ""
        showPile s rest = showHex s ('_' : rest)

newtype PackedTableaux = PackedTableaux { unpackTableaux :: Word16 } deriving (Bounded, Eq, Ix, Ord)

instance Show PackedTableaux where
    showsPrec _ (PackedTableaux p) s = foldr (\pile-> showHex (p `nibbleAt` pile)) s [0..3]

initialTableaux :: [[Card]] -> InitialTableaux
initialTableaux piles = InitialTableaux (listArray (0, 3) (map makePile piles)) where
    makePile pile = let (PackedStack pile') = foldl pushStack (PackedStack 0) pile in pile'

packTableaux :: InitialTableaux -> PackedTableaux
packTableaux (InitialTableaux piles) = PackedTableaux (foldrArray (\ pile prev -> prev `shiftL` 4 .|. pileHeight pile) 0 piles) where
    pileHeight pile = fromIntegral $ 16 - (countLeadingZeros pile `div` 4)

popCard :: InitialTableaux -> PackedTableaux -> Int -> Maybe (PackedTableaux, Card)
popCard (InitialTableaux initial) (PackedTableaux current) pile = if offset == 0 then Nothing else Just (new, card) where
    offset = fromIntegral (current `nibbleAt` pile)
    new = PackedTableaux (current - bit (pile * 4))
    card = fromRank ((initial ! pile) `nibbleAt` (offset - 1))

data Position = Position { currentStack :: !Stack, currentTableaux :: !PackedTableaux } deriving (Show)

instance Ord Position where
    compare = comparing $ \ position -> (currentTableaux position, stack (currentStack position))

instance Eq Position where
    a == b = compare a b == EQ

moves :: InitialTableaux -> Position -> [Position]
moves initial (Position { currentStack, currentTableaux }) = mapMaybe playFrom [0..3] where
    playFrom pile = do
        (tableaux', card) <- popCard initial currentTableaux pile
        stack' <- play currentStack card
        return $! Position { currentStack = stack', currentTableaux = tableaux' }

stacks :: InitialTableaux -> PackedTableaux -> [Position]
stacks initial startTableaux = go (Set.singleton initialPosition) where
    initialPosition = Position { currentStack = emptyStack, currentTableaux = startTableaux }

    insertIfBetter x (y : rest) | currentTableaux x == currentTableaux y
        = if score (currentStack x) > score (currentStack y) then x : rest else y : rest
    insertIfBetter x rest = x : rest

    hasMoves move (todo, _) = (move : todo, True)

    visit position (todo, finished) = (,) todo' $! finished' where
        (todo', found) = foldr hasMoves (todo, False) (moves initial position)
        finished' = if found then finished else insertIfBetter position finished

    go todo = if Set.null todo then [] else finished ++ go (Set.fromList todo') where
        (todo', finished) = foldr visit ([], []) (Set.toAscList todo)

data BestStack = BestStack
    { stackScore :: !Score
    , tableaux :: !PackedTableaux
    } deriving (Show)

bestStacks :: InitialTableaux -> Array PackedTableaux [BestStack]
bestStacks initial = runSTArray $ newGenArray (minBound, maxBound) (return . initialize) where
    -- once the piles are all empty, there are no more moves to make
    initialize (PackedTableaux 0) = []
    initialize current = traceShow (current, bestFirst) $ map extract bestFirst where
        bestFirst = sortBy (comparing orderHeuristic) (stacks initial current)
        -- try the highest scores first, but in case of ties, try the smallest
        -- stacks first, with the idea that playing fewer cards leaves more
        -- opportunities for high scores later
        orderHeuristic (Position position _) = (Down (score position), stack position)
        extract (Position position tableaux) = BestStack { stackScore = score position, tableaux }

search :: Array PackedTableaux [BestStack] -> PackedTableaux -> Score
-- should produce the same score as:
-- search edges current = foldr max 0 [ stackScore s + search edges (tableaux s) | s <- edges ! current ]
search edges start = runST $ do
    scores <- newArray (minBound, maxBound) 0
    parents <- newArray (minBound, maxBound) 0
    writeArray scores start 1
    writeArray parents start (unpackTableaux start)
    go scores parents [start]
    finalScore <- readArray scores minBound
    return (finalScore - 1)
    where
        go :: STUArray s PackedTableaux Score -> STUArray s PackedTableaux Word16 -> [PackedTableaux] -> ST s ()
        go _ _ [] = return ()
        go scores parents (cur:todo) = do
            curBest <- readArray scores cur
            todo' <- forM (edges ! cur) $ \ next -> do
                let newScore = curBest + stackScore next
                nextBest <- readArray scores (tableaux next)
                if newScore > nextBest
                    then do
                        when (tableaux next == minBound) (traceM (" *** HIGH SCORE " ++ show (newScore - 1)))
                        writeArray scores (tableaux next) newScore
                        writeArray parents (tableaux next) (unpackTableaux cur)
                        return (Just (tableaux next))
                    else return Nothing
            go scores parents (catMaybes todo' ++ todo)

main :: IO ()
main = do
    guard $ and playTests
    let initial = initialTableaux $ replicate 4 $ map fromLabel "KQJT98765432A"
    let memo = bestStacks initial
    print $ search memo (packTableaux initial)
