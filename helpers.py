from numpy import dot
from numpy.linalg import norm
import re
#from spellchecker import SpellChecker

#spell = SpellChecker()

def correct_misspell(word_list):
    """
        find misspells from a word list and fix the errors
            word_list: list - a list of words
            return: list - a fixed list of words
    """

    result_list = []
    for word in word_list:
        correct_word = spell.correction(word)
        if (correct_word != '' and word != "'s" and len(word) > 3): # avoid to fix the possession and short words
            result_list.append(correct_word)
        else:
            result_list.append(word)

    return result_list

def remove_emojis(text):
    """
        remove emojis from text (Karim Omaya from stackoverflow.com)
            text: string - a given text
            return: text - a text without emojis
    """
    
    emoj = re.compile('['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002500-\U00002BEF'  # chinese char
        u'\U00002702-\U000027B0'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        u'\U0001f926-\U0001f937'
        u'\U00010000-\U0010ffff'
        u'\u2640-\u2642' 
        u'\u2600-\u2B55'
        u'\u200d'
        u'\u23cf'
        u'\u23e9'
        u'\u231a'
        u'\ufe0f'  # dingbats
        u'\u3030'
                      ']+', re.UNICODE)
    return re.sub(emoj, '', text)


def check_intersection(set_a, set_b):
    """
        check the intersection between 2 sets
            set_a: set
            set_b: set
            return: boolean
    """

    common_set = set_a.intersection(set_b)
    if (len(common_set) == 0): return False
    return True # 2 sets have the common items


def custom_dot(A, B):
    score = 0
    
    try:
        score = (sum(a*b for a, b in zip(A, B)))
    except:
        pass
    
    return score


def cosine_similarity(a, b):
    score = 0
    
    try:
        score = custom_dot(a, b) / ( (custom_dot(a, a) **.5) * (custom_dot(b, b) ** .5) )
    except:
        pass

    return score

def cosine_similarity_numpy(a, b):
    score = 0
    try:
        score = dot(a, b)/(norm(a)*norm(b))
    except:
        pass

    return score

def tensor_to_range01(A, batch_size, height, width):
    """
        convert tensor A to range(0, 1)
        A: tensor
        return: AA - tensor in range(0,1)
        
    """
    
    AA = A.view(A.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(batch_size, height, width)

    return AA


