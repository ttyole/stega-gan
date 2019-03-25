import os
import argparse
import numpy as np

from models.utility.stc import STC
from models.utility.get_image import read_pgm

def bitlist2str(l):
    l_str = map(lambda x: str(int(x)),l)
    s = ''.join([chr(int(''.join(l_str[x:x+8]),base=2)) for x in range(0,len(l),8)])
    return s

def main():
    s_description = (
        "Take a STEGOFILE image as input, use the KEY to retreive the message embeded inside "
        "and print the extracted message to stdout as an ASCII string using utf8 encoding "
        "or write it bitwise to the file EXTRACTEDMSGFILE if given.")

    s_stegofile = ("Path to the STEGO image file conteining a embeded message.")

    s_key = (
        "KEY to retreive the embeded message. Either a string or a path to the file "
        "containing the key")

    s_extractedmsgfile = (
        "Path to the file in which the extracted message will be writen. If omited the "
        "extracted message will be printed to stdout.")

    parser = argparse.ArgumentParser(description=s_description)
    parser.add_argument("STEGOFILE", help=s_stegofile)
    parser.add_argument("KEY", help=s_key)
    parser.add_argument("EXTRACTEDMSGFILE", default=None, nargs='?', help=s_extractedmsgfile)
    args = parser.parse_args()

    # Prepare stego image array
    if not os.path.isfile(args.STEGOFILE):
        print("'"+args.STEGOFILE+"' is not a valide file")
        return
    stego = read_pgm(args.STEGOFILE)
    c_stego = list(np.reshape(stego,np.size(stego)))

    # Prepare decryption key
    try:
        c_key = [int(s) for s in args.KEY.split('|')]
        assert len(c_key) == 2
    except:
        try:
            f = open(args.KEY,'r')
            c_key = [int(s) for s in f.read().split('|')]
            assert len(c_key) == 2
        except:
            print("Not a valid KEY")
            return

    c_extractedmsg = STC.extract(c_stego,c_key)
    extractedmsg = bitlist2str(c_extractedmsg)

    if not args.EXTRACTEDMSGFILE is None:
        with open(args.EXTRACTEDMSGFILE,'wb') as f:
            f.write(extractedmsg)
    else:
        print(extractedmsg.decode('utf8'))

if __name__ == '__main__':
    main()
