alphabet = "abcdefghijklmnopqrstuvwxyz"
vowels = "aeoiu"

def english_to_piglattin( english ):

    piglattin = ""

    in_word = False
    is_first = False
    start = None
    for char in english:
        if not in_word:
            if char in alphabet + alphabet.upper():
                in_word = True

                if char in vowels + vowels.upper():
                    start = None
                    piglattin += char
                else:
                    start = char
                    is_first = True
            else:
                piglattin += char
        else: #if in_word
            if char in alphabet + alphabet.upper():
                if is_first:
                    is_first = False
                    if start in alphabet.upper():
                        piglattin += char.upper()
                    else:
                        piglattin += char
                else:
                    piglattin += char
            else:
                in_word = False
                is_first = False
                if start:
                    piglattin += start.lower() + "ay" + char
                else:
                    piglattin += "yay" + char
    
    #end of sentence needs done as well.
    if in_word:
        if start:
            piglattin += start.lower() + "ay"
        else:
            piglattin += "yay"
    return piglattin


def main():
    used_englishes = []
    with open( "spa.csv", "rt" ) as fin:
        with open( "pig_lattin.csv", "wt" ) as f_out:
            f_out.write( "English,Piglattin\n" )
            for line in fin:
                english = line.split( "\t" )[0]
                english = english.replace( ",", " " )

                if english not in used_englishes:
                    used_englishes.append(english)
                    
                    piglattin = english_to_piglattin( english )

                    f_out.write( f"{english},{piglattin}\n" )

if __name__ == '__main__':
    main()

    # print( english_to_piglattin( "I am not a potato." ) )
    # print( english_to_piglattin( "I am not a potato" ) )
    # print( english_to_piglattin( "I like chicken." ) )
    # print( english_to_piglattin( "Do you know your a b c's?" ) )
    # print( english_to_piglattin( "My name is Joshua." ) )