import sys

def is_pal(input):
  if not input:
    return False
  start = 0
  end = len(input)-1
  while start<=end:
    if input[start]!=input[end]:
      return False
    start+=1
    end-=1
  return True

def r_is_pal(input):
  if len(input)<=1:
    return True
  return input[0]==input[-1] and r_is_pal(input[1:-1])

if __name__=="__main__":
  print r_is_pal(sys.argv[1])
 
