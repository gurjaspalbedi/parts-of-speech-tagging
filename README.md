# Parts of Speech Tagging

Parts of Speech Tagging using Naive Bayes, Viterbi and MCMC with Gibbs sampling

The following three approaches were used for part of speech tagging:

1. Naïve Bayes method
2. Viterbi algorithm
3. MCMC with Gibbs sampling


As a first step we train the data, and store the emission, transition probabilities in a dictionary.

The dictionaries store:

1. Probability of word given speech

    <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;$$P(W|S)$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;$$P(W|S)$$" title="\large $$P(W|S)$$" /></a>

2. Transition probabilities P(Sn|Sn-1) and P(Sn|Sn-1, Sn-2)

    <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;$$P(S_{n}|S_{n-1})&space;\,\,\,\,&space;P(S_{n}|S_{n-1},S_{n-2})$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;$$P(S_{n}|S_{n-1})&space;\,\,\,\,&space;P(S_{n}|S_{n-1},S_{n-2})$$" title="\large $$P(S_{n}|S_{n-1}) \,\,\,\, P(S_{n}|S_{n-1},S_{n-2})$$" /></a>

The above probabilities are stored to use further for the algos.

**NAÏVE BAYES:** Generates most likely tag sequence using naive bayes inference.
Here each part of speech is considered independent of the other

Formulation: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;$$P(S|W)&space;=&space;max{P(W|S)\,&space;P(S)&space;\over&space;P(W)}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\large&space;$$P(S|W)&space;=&space;max{P(W|S)\,&space;P(S)&space;\over&space;P(W)}$$" title="\large $$P(S|W) = max{P(W|S)\, P(S) \over P(W)}$$" /></a>


As P(W) is constant, we ignore it and assign the tag with which we get maximum probability

**VITERBI ALGORITHM:**
Viterbi is based on Dynamic programming for part of speech tagging.

Formulations of the problem:

Part of speech tagging is very common application of Viterbi algorithm. In parts of speech tagging we can only observe the words and we need to determine the Parts of speech of that word. Here the words are observed variables and hidden variables are the parts of speeches.
For Viterbi we learned the transition probabilities and emission probabilities from the training file. Then applied that trained parameters to the testing inputs. This problem can also be solved by using trigram transition probabilities, but we used the just bigram to keep the things simple.

Viterbi is defined as:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;$$V(k&space;,&space;v)&space;=&space;max{&space;(V(k-1,&space;u)&space;*&space;q(v&space;|&space;u)&space;*&space;e(x,&space;v))&space;}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\large&space;$$V(k&space;,&space;v)&space;=&space;max{&space;(V(k-1,&space;u)&space;*&space;q(v&space;|&space;u)&space;*&space;e(x,&space;v))&space;}$$" title="\large $$V(k , v) = max{ (V(k-1, u) * q(v | u) * e(x, v)) }$$" /></a>

Where k = sequence of length k
Given sequence ending with  part of speech v (for length k)
The part of speech u, we compute for all parts of speeches we have
q = the transition probability from u to v
e(x , v) = the emission probability for given word(x) and part of speech(v)

Posterior is calculated as =

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;$$P(S_{1})*{P(S_{2}|S_{1})P(S_{3}|S_{2})…P(S_{n}|S_{n-1})}{&space;P(W_{1}|S_{1})…P(W_{n}|S_{n})}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\large&space;$$P(S_{1})*{P(S_{2}|S_{1})P(S_{3}|S_{2})…P(S_{n}|S_{n-1})}{&space;P(W_{1}|S_{1})…P(W_{n}|S_{n})}$$" title="\large $$P(S_{1})*{P(S_{2}|S_{1})P(S_{3}|S_{2})…P(S_{n}|S_{n-1})}{ P(W_{1}|S_{1})…P(W_{n}|S_{n})}$$" /></a>

Where s denotes the part of speeches and W denotes words

**How program works:**  We have used the list of dictionaries to represent the Viterbi table. The key of the dictionary part of speech we are currently calculating for. The index of the Viterbi list means the word for which we plan to find the part of speech. For given part of speech and word we calculate based on all the parts of speeches and take the maximum value of all and then this value is assigned to cell of the Viterbi table (for given part of speech and word). The path this algorithm take is store in another dictionary and at each step new
part of speech is added to the Viterbi path. At the end we compute the maximum and the path for this maximum part of speech is returned, which is the path our Viterbi algorithm takes.
Design Decision: Everything is changed to lower case to make the comparisons case insensitive. The transition counts are store in one dictionary using which the transition probabilities are calculated and emission counts are stored in different dictionary from which the emission probabilities are calculated.

Results of this evaluation on bc.test file were:

1. With grammar function:

    `a.     Sentence accuracy: 60.65%`
    `b.     Word accuracy: 96.09%`

2. Without grammar function:

    `a.     Sentence accuracy: 54.45%`
    `b.     Word accuracy: 95.05%`

**MCMC GIBBS SAMPLING:**
Here the model assumes a more complicated Bayes net with every part of speech depending on the previous two parts
of speech
Algo/Formulation:

1. Assign a random set of parts of speech to each word, and then choose each word and assign it all the 12 parts of speech. For each word we compute the posterior given by the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;$$P(S_{i}|S-S_{i},&space;W)&space;=&space;(P(S_{1})P(W_{1}|S_{1})P(S_{2}|S_{1})P(S_{3}|S_{1},S_{2})….P(S_{n}|S_{n-1},S_{n-2})$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\large&space;$$P(S_{i}|S-S_{i},&space;W)&space;=&space;(P(S_{1})P(W_{1}|S_{1})P(S_{2}|S_{1})P(S_{3}|S_{1},S_{2})….P(S_{n}|S_{n-1},S_{n-2})$$" title="\large $$P(S_{i}|S-S_{i}, W) = (P(S_{1})P(W_{1}|S_{1})P(S_{2}|S_{1})P(S_{3}|S_{1},S_{2})….P(S_{n}|S_{n-1},S_{n-2})$$" /></a>


2. Rearranging the above terms:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;$$P(S_{1})*{P(S_{2}|S_{1})P(S_{3}|S_{2})…P(S_{n}|S_{n-1})}*{P(S_{3}|S_{1},S_{2})….P(S_{n}|S_{n-1},S_{n-2})}{P(W_{1}|S_{1})…P(W_{n}|S_{n})}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\large&space;$$P(S_{1})*{P(S_{2}|S_{1})P(S_{3}|S_{2})…P(S_{n}|S_{n-1})}*{P(S_{3}|S_{1},S_{2})….P(S_{n}|S_{n-1},S_{n-2})}{P(W_{1}|S_{1})…P(W_{n}|S_{n})}$$" title="\large $$P(S_{1})*{P(S_{2}|S_{1})P(S_{3}|S_{2})…P(S_{n}|S_{n-1})}*{P(S_{3}|S_{1},S_{2})….P(S_{n}|S_{n-1},S_{n-2})}{P(W_{1}|S_{1})…P(W_{n}|S_{n})}$$" /></a>

1. The above terms are marginalised and the probability with each tag ie <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;$P(S_{i}&space;=&space;Noun)$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\large&space;$P(S_{i}&space;=&space;Noun)$" title="\large $P(S_{i} = Noun)$" /></a> and so on.
2. Next we randomly assign the tag each of which has a weight in proportion to its probability.
3. Assign the picked parts of speech for the word and use this modified value for rest of the calculations
4. We do this for all the words once, and then create a sample. This is done over a few hundred to thousand iterations.
5. The first few samples are discarded and the from the rest the tag that occur the most for a word is assigned as the final tag.
   
Results of this evaluation on bc.test file were:
1.	With grammar function:
`a.	Sentence accuracy: 94.51%`
`b.	Word accuracy: 52.21%`

2.	Without grammar function:
`a.	Sentence accuracy: 93.4%`
`b.	Word accuracy: 45.5%`

**Design decisions:**
Initial Sample: To use the sample that was generated by the Naïve bayes model. (The burn in rate and iterations are therefore kept low here as we saw by observation that the model converged very soon by doing this)

Number of iterations: The number of iterations for an arbitrarily taken initial sample are obviously more than for a more informed initial sample(like the one I got from the naïve bayes model). It is observed that it converges fairly fast in the latter case. For eg: a 50 iterations

Log probabilities: We encountered problems of the probabilities getting extremely small and a result becoming zero during MCMC. This was solved by taking their log probabilities

Overall Design decisions:

1. New word handling:
    - The new words are given certain probabilities according to the rules defined in the
    grammar_rules(these probabilities were arrived at empirically by running the code a few times)
    - Certain suffixes have a higher probability of being a certain word, so the new words were given
    probability according to this
    - Prefixes were avoided because there was a higher chance of mis tagging (from observation)
    - The list of suffixes I got was from [here](https://web2.uvcs.uvic.ca/elc/sample/beginner/gs/gs_55_1.htm)
    - If it didn’t fit any of the new conditions it was checked if it was a number, otherwise it was assigned a noun with a certain probability which was again derived empirically.

2.	Data structure
    - Dictionary – we used dictionaries mostly because of the easy access to their values and constant time for fetching
    - Hashing of the already computed probabilities

Results:
As a result it was observed that there was a substantial increase in accuracy as a result of using the grammar
function on bc.train set
Without the **grammar_rules function**(the code has to be modified for this a bit):
So far scored 2000 sentences with 29442 words.
| Type | Words Correct | Sentences Correct |
|:-----------|------------:|:------------:|
| Ground Truch       |        100.00% |     100.00%     |
| Simple     |      91.51% |    36.35%    |
| HMM       |        95.05% |     54.45%     |
| Complex         |          93.41% |      45.85%      |


**With the grammar_rules function:**
| Type | Words Correct | Sentences Correct |
|:-----------|------------:|:------------:|
| Ground Truch       |        100.00% |     100.00%     |
| Simple     |      93.79% |    45.95%    |
| HMM       |        96.09% |     60.65%     |
| Complex         |           94.51% |      52.20%      |


One interesting observation is that even though the MCMC model is more thorough in that it takes dependencies from the previous to previous part of speech, it still has lower probability than HMM. My suspection is that it happens because of overfitting on the training set. Also I’ve learnt that Viterbi usually gives a highest accuracy of 97% using its trigram model

Calculation of the posterior and observations:
The posterior was calculated according to the formulae described above for each model on the solution provided by the other model.
It can be easily observed that Viterbi and Naïve Bayes return the highest value of their probabilities for their own solutions,
however this may not always be the case with MCMC(when it doesn’t converge sometimes) because it is a probabilistic model.

Here is an example:

![Result](https://github.com/gurjaspalbedi/parts-of-speech-tagging/blob/master/resuls.JPG?raw=true)

Here we notice that MCMC assigns the highest probability to the HMM, this can be however fixed though by continuing
the iterations to a larger number.
