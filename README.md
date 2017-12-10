## 2016 Election Project

Paige Haring

University of Pittsburgh

peh40@pitt.edu

The 2016 presidential election in the United States provides a unique opportunity to analyze gender-bias in politics because Hillary Clinton became the first woman presidential nominee of a major political party in the US. Studies have shown that this gender-bias has a real affect on voter patterns and the success of political politicians, and one of the ways this bias can be expressed is in the way we refer to female candidates (Fracchiolla, 2011). This project uses debate transcripts from the 25 debates of the 2016 presidential election to analyze the referring expressions used for and by the candidates.

You can find my visitor's log [here](https://github.com/Data-Science-for-Linguists/Shared-Repo/blob/master/todo10_visitors_log/visitors_log_paige.md).

### Directory
- [Project Plan](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/project_plan.md)
- [Progress Report](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/progress_report.md)
- [Data](https://github.com/Data-Science-for-Linguists/2016-Election-Project/tree/master/data)
  - [Debate Transcripts and CSVs](https://github.com/Data-Science-for-Linguists/2016-Election-Project/tree/master/data/Debates)
  - [Lists](https://github.com/Data-Science-for-Linguists/2016-Election-Project/tree/master/data/Lists)
- [License](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/LICENSE.md)
- [License Notes](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/LICENSE_notes.md)
- [Data Processing](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/data_processing.ipynb)
- [NER Annotation](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/NER_annotating.ipynb)
- [Analysis](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/analysis.ipynb)
- [Presentation Slides](https://github.com/Data-Science-for-Linguists/2016-Election-Project/blob/master/2016_Election_Project_Presentation.pdf)
### Dataset
The data for this project is included in the data/Debates folder. The transcripts of the files are all saved as text files in the data/Debates/transcripts folder. These transcripts are from [The American Presidency Project at UC Santa Barbara](http://www.presidency.ucsb.edu/debates.php) with citations below.

In the data/Debates/csv folder, I have included a comma separated value file for each debate transcript where the values are the date of the debate, the type of debate (general, Democratic primary, or Republican primary), the speaker, the utterance, and NER annotation of the utterance in the form of a tree in the last value. In the data/Lists/ folder, there are text files containing the manually pruned list of all of the entities fount in the transcripts. The file relevant_people.txt is this trimmed list. The file linked.txt is a text file used to link all of the different ways a person is referred to, to that person, in order to create a dictionary later on.

### NER Trees
The NER trees in each csv are set up as follows: The label of the tree represents the speaker of the utterance. The label for each of the tagged named entities within the tree is used to link the annotation to the person it is referring to.

### Citations
My uses for this data fall under the terms of fair use, as I am transforming them from their original transcript form and adding linguistic annotation and analysis for scholarship purposes. The transcripts of these debates can be found from other sources as well.

Fracchiolla, B. (2011). Politeness as a strategy of attack in a gendered political debate – The
Royal-Sarkozy debate. Journal of Pragmatics, 43, 2480-2488

Presidential Candidates Debates: "Presidential Debate at the University of Nevada in Las Vegas," October 19, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=119039.

Presidential Candidates Debates: "Presidential Debate at Washington University in St. Louis, Missouri," October 9, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=119038.

Presidential Candidates Debates: "Presidential Debate at Hofstra University in Hempstead, New York," September 26, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=118971.

Presidential Candidates Debates: "Vice Presidential Debate at Longwood University in Farmville, Virginia," October 4, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=119012.

Presidential Candidates Debates: "Democratic Candidates Debate in Brooklyn, New York," April 14, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=116995.

Presidential Candidates Debates: "Democratic Candidates Debate in Miami, Florida," March 9, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=112719.

Presidential Candidates Debates: "Democratic Candidates Debate in Flint, Michigan," March 6, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=112718.

Presidential Candidates Debates: "Democratic Candidates Debate in Milwaukee, Wisconsin," February 11, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111520.

Presidential Candidates Debates: "Democratic Candidates Debate in Durham, New Hampshire," February 4, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111471.

Presidential Candidates Debates: "Democratic Candidates Forum at Drake University in Des Moines, Iowa," January 25, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=123351.

Presidential Candidates Debates: "Democratic Candidates Debate in Charleston, South Carolina," January 17, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111409.

Presidential Candidates Debates: "Democratic Candidates Debate in Manchester, New Hampshire," December 19, 2015. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111178.

Presidential Candidates Debates: "Democratic Candidates Debate in Des Moines, Iowa," November 14, 2015. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=110910.

Presidential Candidates Debates: "Democratic Candidates Debate in Las Vegas, Nevada," October 13, 2015. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=110903.

Presidential Candidates Debates: "Republican Candidates Debate in Miami, Florida," March 10, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=115148.

Presidential Candidates Debates: "Republican Candidates Debate in Detroit, Michigan," March 3, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111711.

Presidential Candidates Debates: "Republican Candidates Debate in Houston, Texas," February 25, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111634.

Presidential Candidates Debates: "Republican Candidates Debate in Greenville, South Carolina," February 13, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111500.

Presidential Candidates Debates: "Republican Candidates Debate in Manchester, New Hampshire," February 6, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111472.

Presidential Candidates Debates: "Republican Candidates Debate in Des Moines, Iowa," January 28, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111412.

Presidential Candidates Debates: "Republican Candidates Debate in North Charleston, South Carolina," January 14, 2016. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111395.

Presidential Candidates Debates: "Republican Candidates Debate in Las Vegas, Nevada," December 15, 2015. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=111177.

Presidential Candidates Debates: "Republican Candidates Debate in Milwaukee, Wisconsin," November 10, 2015. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=110908.

Presidential Candidates Debates: "Republican Candidates Debate in Boulder, Colorado," October 28, 2015. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=110906.

Presidential Candidates Debates: "Republican Candidates Debate in Simi Valley, California," September 16, 2015. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=110756.

Presidential Candidates Debates: "Republican Candidates Debate in Cleveland, Ohio," August 6, 2015. Online by Gerhard Peters and John T. Woolley, The American Presidency Project. http://www.presidency.ucsb.edu/ws/?pid=110489.
