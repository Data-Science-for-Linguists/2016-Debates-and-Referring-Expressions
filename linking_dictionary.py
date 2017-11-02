import pickle

f = open('/Users/Paige/Documents/Data_Science/2016-Election-Project/data/Lists/relevant_people.txt')
people = f.readlines()
f.close()

link_dict = {}

for person in people:
    name = input(person)
    link_dict[person] = name
