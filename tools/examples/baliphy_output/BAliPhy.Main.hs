{-# LANGUAGE ExtendedDefaultRules #-}
module Main where
import Bio.Alignment
import Bio.Alphabet
import Bio.Sequence
import IModel
import MCMC
import Probability
import Probability.Random
import SModel
import SModel.Parsimony
import Tree
import Tree.Newick
import qualified Data.IntMap as IntMap
import qualified Data.JSON as J
import qualified Data.Text.IO as T
import Probability.Logger
import System.Environment
import System.FilePath

sample_smodel  alpha = do {kappaPur <- sample (logNormal (log 2) 0.25)
;kappaPyr <- sample (logNormal (log 2) 0.25)
;pi <- sample (symmetricDirichletOn (getLetters alpha) 1)
;let {result = tn93' alpha kappaPur kappaPyr pi}
;let {loggers = ["tn93:kappaPur" %=% kappaPur, "tn93:kappaPyr" %=% kappaPyr, "tn93:pi" %=% pi]}
;return (result, loggers)
}

sample_imodel  topology = do {rate <- sample (logLaplace (negate 4) 0.707)
;mean_length <- sample (shifted_exponential 10 1)
;let {result = IModel.rs07 rate mean_length topology}
;let {loggers = ["rs07:rate" %=% rate, "rs07:mean_length" %=% mean_length]}
;return (result, loggers)
}

sample_scale  = sample (gamma 0.5 2)

sampleTree taxa = sample (uniformLabelledTree taxa (gamma 0.5 (1 / intToDouble (length taxa))))

model sequenceData logParamsTSV logTree [logA] = do {let {taxa = getTaxa sequenceData}
;
;
;tree <- sampleTree taxa
;(indelRates, log_indelRates) <- do {sigma <- sample (logLaplace (negate 3) 1)
;xs <- sample (iidMap (getUEdgesSet tree) (logNormal 0 1))
;let {result = fmap (\x_4 -> x_4 Prelude.** sigma) xs}
;let {loggers = ["sigma" %=% sigma]}
;return (result, loggers)
}
;let {indelTree = addBranchRates indelRates tree}
;
;let {tlength = treeLength tree}
;scale1 <- sample_scale
;addMove 2 (scaleGroupsSlice [scale1] (branchLengths tree))
;addMove 1 (scaleGroupsMH [scale1] (branchLengths tree))
;(smodel, log_smodel) <- sample_smodel dna
;(imodel, log_imodel) <- sample_imodel tree
;
;let {sequence_lengths = getSequenceLengths sequenceData}
;(alignment, properties_A) <- sampleWithProps (phyloAlignment indelTree imodel scale1 sequence_lengths)
;properties <- observe sequenceData (phyloCTMC tree alignment smodel scale1)
;
;let {alignment_length = alignmentLength alignment}
;let {num_indels = totalNumIndels alignment}
;let {total_length_indels = totalLengthIndels alignment}
;let {prior_A = ln (probability properties_A)}
;let {anc_alignment = toFasta (prop_anc_seqs properties)}
;let {substs = parsimony tree (unitCostMatrix dna) (sequenceData, alignment)}
;let {p1_loggers = ["|A|" %=% alignment_length, "#indels" %=% num_indels, "|indels|" %=% total_length_indels, "prior_A" %=% prior_A, "likelihood" %=% ln (prop_likelihood properties), "#substs" %=% substs]}
;
;let {alignmentLengths = [alignment_length]}
;let {scale = scale1}
;let {loggers = ["indelRates" %>% log_indelRates, "|T|" %=% tlength, "scale1" %=% scale1, "scale1*|T|" %=% (scale1 * tlength), "S1" %>% log_smodel, "I1" %>% log_imodel, "P1" %>% p1_loggers, "scale" %=% scale, "scale*|T|" %=% (scale * tlength), "|A|" %=% sum alignmentLengths, "#indels" %=% sum [num_indels], "|indels|" %=% sum [total_length_indels], "#substs" %=% sum [substs], "prior_A" %=% sum [prior_A]]}
;
;
;addLogger $ logParamsTSV loggers
;
;addLogger $ logTree (addInternalLabels (scaleBranchLengths scale tree))
;
;addLogger $ (every 10 $ logA anc_alignment)
;
;return loggers
}

main = do {[directory] <- getArgs
;sequenceData <- mkUnalignedCharacterData dna <$> loadSequences "5d-clustalw.fasta"
;
;logParamsTSV <- tsvLogger (directory </> "C1.log") ["iter"]
;
;logTree <- treeLogger (directory </> "C1.trees")
;
;logA <- alignmentLogger (directory </> "C1.P1.fastas")
;
;mymodel <- makeMCMCModel $ model sequenceData logParamsTSV logTree [logA]
;
;runMCMC 200000 mymodel
}
