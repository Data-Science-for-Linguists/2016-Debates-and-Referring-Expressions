import nltk

txt= "Please welcome Secretary Hillary Clinton, Senator Bernie Sanders, and Governor Martin O'Malley."
sent = nltk.sent_tokenize(txt)
words = [nltk.word_tokenize(s) for s in sent]
pos = [nltk.pos_tag(sent) for sent in words]
chunk = list(nltk.ne_chunk_sents(pos))
tree = chunk[0]

#Add an extra spot for the title
tree[3].append(tree[3][1])
#Move the current leaves down
tree[3][1]=tree[3][0]
#Add the title to the subtree
tree[3][0]=tree[2]
#Get rid of the title from the rest of the tree
tree.remove(tree[2])
print(tree)
