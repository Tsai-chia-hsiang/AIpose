import json 
def clean(rawfile, cleanfile):
    label = []

    with open(rawfile, "r", encoding="utf-8") as f:
        for li in f.readlines():
            l = li.strip()
            if '/' in l:
                u = l.split("/")
                l = f"{u[1].strip()} {u[0].strip()}"
            if '_' in l:
                l = l.replace('_'," ")
            label.append(l)

    with open(cleanfile, "w+") as f:
        j = {}
        for i,l in enumerate(label):
            j[i] = l
        json.dump(j,f,ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    clean("label.txt", "format_label.json")       
