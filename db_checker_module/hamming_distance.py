import re

def hamming_distance(string1: str, string2:str)->int:
    distance = 0
    string1 = re.sub('\W+','', string1).lower()
    string2 = re.sub('\W+','', string2).lower()
    for char1, char2 in zip(string1, string2):
        if (char1 != char2):
            print(f"char1:{char1} char2:{char2}")
            distance += 1
    return distance

if __name__ == "__main__":
    print(hamming_distance("I Won't Give Up", "i wont give up"))