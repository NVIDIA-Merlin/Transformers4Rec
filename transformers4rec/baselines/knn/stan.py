from _operator import itemgetter
from math import sqrt, exp
import random
import time

import numpy as np
import pandas as pd
from math import log10
from datetime import datetime as dt
from datetime import timedelta as td
import math

class STAN:
    '''
    STAN( k,  sample_size=5000, sampling='recent', remind=True, extend=False, lambda_spw=1.02, lambda_snh=5, lambda_inh=2.05 , session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' )
    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    remind : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    extend : string
        Decay function to determine the importance/weight of individual actions in the current session (linear, same, div, log, quadratic). (default: div)
    lambda_spw : string
        Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic). (default: div_score)
    lambda_snh : boolean
        Experimental function to give less weight to items from older sessions (default: False)
    lambda_inh : boolean
        Experimental function to use the dwelling time for item view actions as a weight in the similarity calculation. (default: False)
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__( self, k, sample_size=5000, sampling='recent', remind=True, extend=False, lambda_spw=1.02, lambda_snh=5, lambda_inh=2.05 , session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):
       
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        
        self.lambda_spw = lambda_spw
        self.lambda_snh = lambda_snh * 24 * 3600
        self.lambda_inh = lambda_inh
        
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        
        self.extend = extend
        self.remind = remind
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()
        self.min_time = -1
        
        self.sim_time = 0
        
    def fit(self, train, test=None, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''            
        self.num_items = train[self.item_key].max()
        
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        index_time = train.columns.get_loc( self.time_key )
        
        session = -1
        session_items = []
        time = -1
        #cnt = 0
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session : time})
                    if time < self.min_time:
                        self.min_time = time
                session = row[index_session]
                session_items = []
            time = row[index_time]
            session_items.append(row[index_item])
            
            # cache sessions involving an item
            map_is = self.item_session_map.get( row[index_item] )
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item] : map_is})
            map_is.add(row[index_session])
            
        # Add the last tuple    
        self.session_item_map.update({session : session_items})
        self.session_time.update({session : time})
        
        if self.sample_size == 0: #use all session as possible neighbors
            print('!!!!! runnig KNN without a sample size (check config)')
        
    def predict_next( self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, timestamp=0, skip=False, type='view'):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
#         gc.collect()
#         process = psutil.Process(os.getpid())
#         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')
        
        if( self.session != session_id ): #new session
            
            if( self.extend ):
                self.session_item_map[self.session] = self.session_items;
                for item in self.session_items:
                    map_is = self.item_session_map.get( item )
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item : map_is})
                    map_is.add(self.session)
                    
                ts = time.time()
                self.session_time.update({self.session : ts})
                
            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()
        
        if type == 'view':
            self.session_items.append( input_item_id )
        
        if skip:
            return
         
        neighbors = self.find_neighbors( self.session_items, input_item_id, session_id, timestamp )
        scores = self.score_items( neighbors, self.session_items, timestamp )
        
        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d( predict_for_item_ids, list(scores.keys()) )
        
        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)
        
        return series 
    
    def vec(self, current, neighbor, pos_map):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        intersection = current & neighbor
        vp_sum = 0
        for i in intersection:
            vp_sum += pos_map[i]
        
        result = vp_sum / len(pos_map)

        return result
    
    def cosine(self, current, neighbor, pos_map):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
                
        lneighbor = len(neighbor)
        intersection = current & neighbor
        
        if pos_map is not None:
            
            vp_sum = 0
            current_sum = 0
            for i in current:
                current_sum += pos_map[i] * pos_map[i]
                if i in intersection:
                    vp_sum += pos_map[i]
        else:
            vp_sum = len( intersection )
            current_sum = len( current )
                
        result = vp_sum / (sqrt(current_sum) * sqrt(lneighbor))
        
        return result
    
    
    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);
    
    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get( item_id ) if item_id in self.item_session_map else set()
        
        
    def most_recent_sessions( self, sessions, number ):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))
            
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample


    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors( self, session_items, input_item_id, session_id, timestamp ):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        possible_neighbors = self.possible_neighbor_sessions( session_items, input_item_id, session_id )
        possible_neighbors = self.calc_similarity( session_items, possible_neighbors, timestamp )
        
        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1] )
        possible_neighbors = possible_neighbors[:self.k]
        
        return possible_neighbors
    
    
    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id )
               
        if self.sample_size == 0: #use all session as possible neighbors
            
            #print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else: #sample some sessions
                         
            if len(self.relevant_sessions) > self.sample_size:
                
                if self.sampling == 'recent':
                    sample = self.most_recent_sessions( self.relevant_sessions, self.sample_size )
                elif self.sampling == 'random':
                    sample = random.sample( self.relevant_sessions, self.sample_size )
                else:
                    sample = self.relevant_sessions[:self.sample_size]
                    
                return sample
            else: 
                return self.relevant_sessions
                        

    def calc_similarity(self, session_items, sessions, timestamp ):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''
        
        pos_map = None
        if self.lambda_spw:
            pos_map = {}
        length = len( session_items )
        
        pos = 1
        for item in session_items:
            if self.lambda_spw is not None: 
                pos_map[item] = self.session_pos_weight( pos, length, self.lambda_spw )
                pos += 1
            
        #print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        items = set(session_items)
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            n_items = self.items_for_session( session )

            similarity = self.cosine(items, set(n_items), pos_map) 
                            
            if self.lambda_snh is not None:
                sts = self.session_time[session]
                decay = self.session_time_weight(timestamp, sts, self.lambda_snh)
                
                similarity *= decay
                            
            neighbors.append((session, similarity))
                
        return neighbors
    
    def session_pos_weight(self, position, length, lambda_spw):
        diff = position - length
        return exp( diff / lambda_spw )
    
    def session_time_weight(self, ts_current, ts_neighbor, lambda_snh):
        diff = ts_current - ts_neighbor
        return exp( - diff / lambda_snh )
            
    def score_items(self, neighbors, current_session, timestamp):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        s_items = set( current_session )
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            n_items = self.items_for_session( session[0] )
            
            pos_last = {}
            pos_i_star = None
            for i in range( len( n_items ) ):
                if n_items[i] in s_items: 
                    pos_i_star = i + 1
                pos_last[n_items[i]] = i + 1
            
            n_items = set( n_items )
            
            for item in n_items:
                
                if not self.remind and item in s_items:
                    continue
                
                old_score = scores.get( item )
                
                new_score = session[1]
                
                if self.lambda_inh is not None: 
                    new_score = new_score * self.item_pos_weight( pos_last[item], pos_i_star, self.lambda_inh )
                
                if not old_score is None:
                    new_score = old_score + new_score
                    
                scores.update({item : new_score})
                    
        return scores
    
    def item_pos_weight(self, pos_candidate, pos_item, lambda_inh):
        diff = abs( pos_candidate - pos_item )
        return exp( - diff / lambda_inh )
    
    def clear(self):
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()
