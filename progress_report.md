## November 26, 2017
- Worked on simple analysis! Created the speaker dataframes
- When I was doing that, I realized there were some issues with my tagger. When I pulled in titles and names that were just completely missed by the tagger, they weren't being tagged correctly even when it worked with some test trees. The issue turned out to be that my mapping function that returned the original tree to the dataframe returned a list whose only element was the tree. That messed up some indexing things later on in later functions, so when I fixed that, my tagger worked a lot better!

### Tasks
- What is the best way to include all data? My master csv doesn't store the trees correctly. I want to include a pickle file, but that might be too big?

## November 19, 2017
- Pulled in missed titles to the trees
- Worked on a function to pull in all missed RE's

## November 2, 2017
- Mapped RE to entity it was referring too
- Really basic analysis

### Tasks
- Fix under-labelling
- More linguistic analysis

## November 1, 2017
- Updated license and justified choice
- Updated progress report
- Cleaned up pipeline
- Made good progress on entity linking in NER trees

## October 29, 2017
- Did rough NER annotation on each data frame
- Set label of each NER tree to the speaker

### Tasks
- Figure out how to link entities
- Look into analysis
- Update project plan

## October 25, 2017
- Processed all of the primary debates (and general for a second time in a different way) and cleaned them up
- Saved all of the clean data frames to csv and included them in data/csv folder
- Reorganized data in folders
- Tried spaCy NER annotation

### Tasks
- NER annotation on all debates
- Figure out how to link entities
- Look into analysis
- Update project plan

## October 24, 2017
- Added the transcript txt files for the primary debates
- Made all data publicly available
- Modified README to explain why I think I can make all of my data public (fair use)
- Looked into NER annotators (nltk's) and tried it out on the first debate's transcript

### Tasks
- Try out other NER annotators
- Figure out how to get the NER annotations into the data frames I made
- Analysis

## October 12, 2017
I cleaned up my debate text files and organized them into data frames, then saved them as CSV files.
The data frames include a column about the date of the debate, which debate it is, the location of the debate, the source url of the transcript, the speaker, the sentence spoken, and any names or referring expressions used in that sentence. Each debate is in a separate data frame and CSV file for now. I did all of the manual RE annotation for the debates.

### Tasks
- Clean up debate 2 (it was a town hall, so some of the questions got formatted in a weird way)
- Decide whether or not to use other speeches and if so, which ones?

## October 1, 2017

I've created my project repo, downloaded the datasets I'll need for my analysis, and developed a project plan. I manually copied and pasted the debated transcripts into text files since there were only four of them. I began poking around the data that I have and realized I have some clean up to do.

### Tasks
- Manually annotate for referring expressions
- Clean up extra data
  - annotations for who is speaking, applause, laughter, etc.
- Reformat text files so they're all uniform
- Decide whether to stick with just debates, or all speeches
