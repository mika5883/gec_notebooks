import argparse
import errant
import pandas as pd
import csv
from tqdm import tqdm

def process_row(row, annotator):
    try: 
        orig = row['corrupt_sent'] if 'corrupt_sent' in row else row['corrupt']
        cor = row['correct_sent'] if 'correct_sent' in row else row['corrected']
        or_p = annotator.parse(orig)
        cor_p = annotator.parse(cor)
        alignment = annotator.align(or_p, cor_p, lev=False)
        edits = annotator.merge(alignment, merging='rules')
        
        edit_list = []
        for e in edits:
            e = annotator.classify(e)
            edit = annotator.import_edit(or_p, cor_p, edit_edit(e))
            edit_list.append(edit.to_m2())
        
        return ('S ' + orig, edit_list)
    except Exception as ex:
        return None



def read_df(tsv_path, sep='\t'):
    return pd.read_csv(tsv_path, sep=sep, quoting=csv.QUOTE_NONE)

def edit_edit(e):
    return e.o_start, e.o_end, e.c_start, e.c_end, e.type

def df2_m2_batched(df, annotator) -> dict:
    m2_dict = {}
    try:
        orig_texts = df['corrupt_sent'] if 'corrupt_sent' in df.columns else df['corrupt']
        cor_texts = df['correct_sent'] if 'correct_sent' in df.columns else df['corrected']
    except KeyError as e:
        raise ValueError("Missing expected columns in TSV") from e

    orig_parsed = list(tqdm(annotator.nlp.pipe(orig_texts, batch_size=64, n_process=4), total=len(df), desc="Parsing corrupt"))
    cor_parsed = list(tqdm(annotator.nlp.pipe(cor_texts, batch_size=64, n_process=4), total=len(df), desc="Parsing corrected"))

    for orig_text, or_p, cor_text, cor_p in tqdm(zip(orig_texts, orig_parsed, cor_texts, cor_parsed), total=len(df), desc="Aligning & classifying"):
        alignment = annotator.align(or_p, cor_p, lev=False)
        edits = annotator.merge(alignment, merging='rules')
        m2_dict['S ' + orig_text] = []
        for e in edits:
            e = annotator.classify(e)
            edit = annotator.import_edit(or_p, cor_p, edit_edit(e))
            m2_dict['S ' + orig_text].append(edit.to_m2())
    return m2_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-inp', type=str, help='TSV file path with corrupt and corrected sents')
    parser.add_argument('-out', type=str, help='where you want to store the data')
    parser.add_argument('-sep', default='\t', type=str, help='what symbol is used to sep data in your file')
    args = parser.parse_args()
    print(args)
    
    annotator = errant.load('ru')

    df = read_df(args.inp, sep=args.sep)
    if 'grammar:' in df.iloc[0]['corrupt']:
        df['corrupt'] = df['corrupt'].map(lambda x: x[9:])
    
    m2_dict = df2_m2_batched(df, annotator)

    with open(f'{args.out}', 'w') as w:
        for k, v in m2_dict.items():
            w.write(k.strip() + '\n')
            for each in v:
                w.write(each.strip() + '\n')
            w.write('\n')
