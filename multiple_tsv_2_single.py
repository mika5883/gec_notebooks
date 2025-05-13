import argparse
import pandas as pd

def main(args):
	# some stuff
	df = pd.read_csv(args.inp, sep=args.sep)
	df['hypotheses'] = df['hypotheses'].map(lambda x: eval(x)[args.index] if len(eval(x)[args.index])!=0 else eval(x)[args.index+1])
	df = df.rename(columns={'hypotheses' : 'corrected'})
	df.to_csv(f'{args.out}', sep=args.sep, index=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-inp', type=str, required=True)
	parser.add_argument('-sep', type=str, default='\t')
	parser.add_argument('-out', type=str)
	parser.add_argument('-index', type=int, default=0)
	args = parser.parse_args()

	main(args)