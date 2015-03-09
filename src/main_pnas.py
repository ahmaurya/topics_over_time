# Copyright 2015 Abhinav Maurya

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from tot import TopicsOverTime
import numpy as np
import pickle

def main():
	datapath = '../../data/pnas/'
	resultspath = '../results/pnas_tot/'
	documents_path = datapath + 'alltitles'
	timestamps_path = datapath + 'alltimes'
	stopwords_path = datapath + 'allstopwords'
	tot_topic_vectors_path = resultspath + 'pnas_tot_topic_vectors.csv'
	tot_topic_mixtures_path = resultspath + 'pnas_tot_topic_mixtures.csv'
	tot_topic_shapes_path = resultspath + 'pnas_tot_topic_shapes.csv'
	tot_pickle_path = resultspath + 'pnas_tot.pickle'

	tot = TopicsOverTime()
	documents, timestamps, dictionary = tot.GetPnasCorpusAndDictionary(documents_path, timestamps_path, stopwords_path)
	par = tot.InitializeParameters(documents, timestamps, dictionary)
	theta, phi, psi = tot.TopicsOverTimeGibbsSampling(par)
	np.savetxt(tot_topic_vectors_path, phi, delimiter=',')
	np.savetxt(tot_topic_mixtures_path, theta, delimiter=',')
	np.savetxt(tot_topic_shapes_path, psi, delimiter=',')
	tot_pickle = open(tot_pickle_path, 'wb')
	pickle.dump(par, tot_pickle)
	tot_pickle.close()

if __name__ == "__main__":
    main()