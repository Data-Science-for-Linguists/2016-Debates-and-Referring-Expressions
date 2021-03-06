# What Did You Call Me?
### Referring Expressions, Titles, and Name-calling in the 2016 US Election

##### Final report
##### Paige Haring

## 0. Background

In the 2007 French Presidential Election, Ségolène Royal made history as the first female candidate to make it to the second round of a presidential election, and therefore the first female candidate to make it to the debate stage (Fracchiolla, 2011). Béatrice Fracchiolla seized this opportunity to observe whether a female presence changed the dynamics of the debate by studying the way Royal and her competitor, Nicolas Sarkozy, addressed each other and looking for evidence of gender-bias. Gender stereotypes, like the idea that females are passive and less competent than males, have a significant impact on voting patterns and women's success in politics. Therefore, being able to distinguish the ways these stereotypes are reinforced through speech can help female candidates overcome them (Fracchiolla, 2011). Fracchiolla noticed that Sarkozy used politeness as a "courteous attack" and referred to Royal with gendered terms like Madame very often in order to remind the audience of Royal's gender and appeal to these gender stereotypes.

## 1. Introduction

Hillary Clinton became the first woman presidential nominee of a major political party in the United States during the 2016 Presidential Election. Her opponent in the general election, Donald Trump, was a business man with no political experience. I thought this election provided another opportunity to look for evidence of gender-bias on the debate stage. Like Fracchiolla, I wanted to analyze the way the candidates referred to each other. While Fracchiolla focused largely on gendered terms of address and pronoun usage, I focused on the use of professional titles versus the use of candidates' first or full names. I wanted to see if there were differences in the ways female and male candidates were addressed, and to see if there were differences in the ways candidates with and without political experience were referred to. If so, what could these differences mean?

## 2. Data Sourcing

The data for this project came from the [University of Santa Barbara's American Presidency Project](http://www.presidency.ucsb.edu/debates.php). I used the 12 transcripts from the republican debates, the 10 transcripts from the democratic debates, and the 3 transcripts from the general debates between Donald Trump and Hillary Clinton as my data. I was able to use this data under the terms of fair use, as I used it for scholarship purposes and modified it significantly. The data can be found [here](https://github.com/Data-Science-for-Linguists/2016-Election-Project/tree/master/data/Debates/transcripts).

## 3. Data Cleaning

There were several important steps involved in making this data usable for the task at hand. This process was documented in the [data processing file](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/data_processing.md) in this repository. The original format of this data was 25 separate text files. My ultimate goal was to create DataFrames for each debate where each row contained information about an utterance in the order those utterances were spoken. This information included the speaker of the utterance, the date of the debate, and the kind of debate it was. First, I had to [split each debate transcript into chunks](data_processing.md#splitting-transcripts-by-speaker) where each chunk included all of the utterances one speaker said at a time, while maintaining the order of these chunks of utterances. Then, I had to [split each of these chunks](data_processing.md#tokenizing-each-speakers-sentences) into individual utterances. After this, my data was in a [more organized format](data_processing.md#reordering-and-naming-columns) and the next step was to do [Named Entity Recognition (NER) annotation](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/NER_annotating.md) on each of these utterances. In order to know how a person was referred to, I first needed to find every instance where that person was mentioned. I did this NER annotation using NLTK's default NE chunker. This resulted in NER trees like this:

![png](images/output_17_0.png)

I wanted the speaker of the utterance to be evident by looking at the tree, so I [mapped the speaker column of the DateFrames](NER_annotating.md#mapping-speaker-to-tree) I created to the head label of each tree. After that, they looked like this:

![png](images/output_19_0.png)

The next step was to link the entities that were tagged in these trees to who they were referring to. I did this by manually creating a large list of relevant people, and [creating a dictionary](NER_annotating.md#creating-dictionary-for-ner-linking) that mapped each way of referring to a person to that person.

### Setbacks

The first issue I ran into using NLTK's chunker was that I specifically wanted to look at titles, and the chunker is designed to exclude titles. Sometimes it does not do this well, and titles are included as part of the entity, but most of the time they were excluded. Here is an example:

![png](images/output_23_0.png)

I had to write [a function](NER_annotating.md#pulling-in-titles) that would pull in the preceding titles.

![png](images/output_84_0.png)

The second issue was that the chunker simply did not tag a lot of entities that should have been tagged. I created a function that would tag these missed entities first by searching for [relevant last names](NER_annotating.md#tagging-missed-entities-last-names). Then again later by looking for [titles and first names](NER_annotating.md#tagging-missed-entities-titles-and-first-names).

That changed my trees from this:

![png](images/output_88_0.png)

to this:

![png](images/output_90_0.png)

then this:

![png](images/output_95_0.png)

There were many other issues with the chunker that I couldn't fix. For example, in the first tree above, the word "Please" was tagged as a person. I did not untag any of the mistakenly tagged "entities" because it was too time consuming. Also, since I created the linking dictionary manually and people were occasionally referred to by their first name, I made the decision to make sure I mapped each candidate's first name to them. This lead to some difficulties in distinguishing between people with the same first name. For example, since Chris Christie was a candidate, I mapped Chris to Chris Christie. This caused problems because Chris Wallace was a moderator. That caused each time Chris Wallace was referred to as Chris to be mistakenly tagged as Chris Christie. In the future, I might try a different NER tool.

## 4. Analysis

I analyzed the way eight candidates referred to each other. I conducted analysis for [Hillary Clinton](analysis.md#hillary-clinton-as-speaker), [Donald Trump](analysis.md#donald-trump-as-speaker), [Bernie Sanders](analysis.md#bernie-sanders-as-speaker), [Ted Cruz](analysis.md#ted-cruz-as-speaker), [Marco Rubio](analysis.md#marco-rubio-as-speaker), [Carly Fiorina](analysis.md#carly-fiorina-as-speaker), and [Ben Carson](analysis.md#ben-carson-as-speaker) as speakers referring to each of the other candidates. I also looked to see how each of these candidates were referred to by any speaker throughout the transcript data. Please see my [analysis script](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/analysis.md) for a more detailed analysis of each individual speaker.

I created six categories of referring expressions: first names, last names, full names, professional title, gendered title, or name-calling.

Originally, I created the "gendered title" category because I believed it would be helpful in looking at gender-bias. Examples of "gendered titles" are Mrs., Miss, Mr., Madam, etc. It was easy for me to distinguish when Hillary Clinton was referred to by a gendered title, because she has multiple professional titles that she could be referred to like Secretary Clinton. The problem I had was deciding how to handle "Mrs. Fiorina" when referring to Carly Fiorina. This is because not only is that a gendered title, but it was also the only professional title that could be used for her because she never held a political office. The same was true for Donald Trump and "Mr.". I decided that in the absence of a political title, Mr. or Mrs. would not be considered gendered.

I also looked at how [Dana Bash and David Muir](analysis.md#moderators) referred to these eight candidates to see if there were differences in how a moderator and a competitor might refer to the same person.

Finally, I looked to see if there were [changes](analysis.md#differences-across-debates) in the way Hillary Clinton and Donald Trump referred to each other across the three general debates.

I have found some interesting results. I don't believe I can make any definite conclusions about gender-bias evident in these transcripts with the data I have, but there are some things that we can observe easily. It is no surprise that the moderators we looked at never refer to the candidates by their first name and almost always refer to them by their professional titles. Interestingly, Dana Bash, a female moderator, used the expression Madam Secretary more often than David Muir did when describing Clinton. When using a professional title for Clinton, Muir did not add gender to the term.

![png](images/output_296_0.png)

![png](images/output_324_0.png)

We also found that out of all of the times Clinton is referred to, she is called by a professional title a smaller percentage of the time than the percentage of the each of the male candidates we looked at. Here is a distribution of the different types of referring expressions used by any speaker in the dataset for the 8 candidates mentioned above.

![png](images/output_92_0.png)

Clinton refers to other candidates by their professional title most often, but refers to Donald Trump by his first name most often.

![png](images/output_147_0.png)

This could have been strategic. Clinton could have been trying to take away some of the power Trump might have had by not referring to him with a title. Trump, who in the first debate called Clinton by a professional title most often, completely stopped doing that by the second in an attempt to match the way Clinton referred to him.

![png](images/output_361_0.png)

It was interesting that Trump called Clinton by a title in the first debate, because in all of the primary debates preceding it, Trump referred to the other candidates involved by their first names most often. His use of a title for her was most likely a strategy to try to get respect from her by being more polite, or to make sure he did not look bad by not using a title. When Clinton made it clear she would only call him by his first name, Trump changed his strategy, because it doesn't help him to refer to her in a way that emphasizes her experience in politics in a respectful way while she refers to him in a less formal way. It makes him seem less powerful.

We should also consider what it could mean to call someone by their full name. In a debate where someone is called by their full name and they are not participating, the speaker must be talking *about* that person because on the one hand, they are not there to address, and on the other hand, using a full name is a specific form of address used to inform everyone of exactly who you are talking about. So what does it mean to be talked about a lot versus not at all? Talking about someone is obviously indicative of how relevant that person is in the campaign, and could be indicative of how much of a threat that person is considered to be. If a candidate talks about another candidate a lot, it could mean that they are relatively confident they will be directly compared to them in the future, meaning the speaker is confident both they and the person they are talking about will continue on into later rounds of the debates. This of course is not the only explanation of these phenomena, but certainly a possible explanation. To use a candidate's full name in a debate where you are both participating could be interpreted as confrontational. The speaker could also be talking about the candidate as if they were not there to address more personally to make them seem like a less powerful presence.

## 5. Conclusion

I would like to continue researching this subject and look into pronoun usage like Fracchiolla did. I would also like to add more information to my trees to somehow include who is being spoken *to*. I also need to do more research regarding what different forms of address typically mean, and I would like to look more into name-calling across the entire election. Overall, I think I have created a good starting tool for other researchers to be able to analyze referring expressions in political-debate discourse.

## 6. References

Fracchiolla, B. (2011). Politeness as a strategy of attack in a gendered political debate – The Royal-Sarkozy debate. Journal of Pragmatics, 43, 2480-2488
