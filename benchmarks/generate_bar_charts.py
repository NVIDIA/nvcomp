#! /usr/bin/env python

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import argparse
import os

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file", type=str, help="the file to read the data from")
  parser.add_argument("plot_directory", type=str, help="the file to read the data from", nargs='*', default='plots')
  args = parser.parse_args()
  df = pd.read_csv(args.input_file)
  df['dataset'] = df['dataset'].str.replace("mortgage-2009Q2-float-columns.bin", "Mortgage Float", regex=False)
  df['dataset'] = df['dataset'].str.replace("mortgage-2009Q2-string-columns.bin", "Mortgage String", regex=False)
  df['dataset'] = df['dataset'].str.replace("mortgage-2009Q2-col0-long.bin", "Mortgage Long", regex=False)
  df['dataset'] = df['dataset'].str.replace("geometrycache.tar", "Geometry", regex=False)
  df['dataset'] = df['dataset'].str.replace("texturecache.tar", "Texture", regex=False)
  df['dataset'] = df['dataset'].str.replace("silesia.tar", "Silesia", regex=False)
  df['dataset'] = df['dataset'].str.replace(" ", "\n", regex=False)

  if not os.path.exists(args.plot_directory):
    os.makedirs(args.plot_directory)

  sns.set(style="whitegrid")
  for interface in ['LL', 'HL']:
    for metric in ['compression_ratio', 'compression_throughput', 'decompression_throughput']:
      plt.figure()
      title_metric = metric.replace('_',' ').title()
      bar = sns.barplot(x='dataset', y=metric, hue='algorithm', data=df[df.interface==interface])
      for container in bar.containers:
        bar.bar_label(container, fmt="%.1f")

      bar.set_yscale("log")
      plt.title(f"{interface.replace('LL', 'Low Level').replace('HL', 'High Level')} {title_metric}")
      plt.xlabel('')
      plt.ylabel(title_metric)
      plt.legend(loc=(1.04,0))
      plt.savefig(f"{args.plot_directory}/{metric}-{interface}.png", bbox_inches='tight')
  plt.show()

if __name__ == "__main__":
  main()
