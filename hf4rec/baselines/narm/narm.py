import numpy as np
import pandas as pd
import theano.tensor as T
import theano
from collections import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams
import time
import sys


class NARM:
    '''
    Code based on work by Li et al., Neural Attentive Session-based Recommendation, CIKM 2017.
    NARM(factors=100, session_key='SessionId', item_key='ItemId')
    
    Popularity predictor that gives higher scores to items with larger support.
    
    The score is given by:
    
    .. math::
        r_{i}=\\frac{supp_i}{(1+supp_i)}
        
    Parameters
    --------
    top_n : int
        Only give back non-zero scores to the top N ranking items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    item_key : string
        The header of the item IDs in the training data. (Default value: 'ItemId')
    support_by_key : string or None
        If not None, count the number of unique values of the attribute of the training data given by the specified header. If None, count the events. (Default value: None)
    
    '''
    
    def __init__(self, factors=100, hidden_units=100, epochs=30, epochs_patience=5, lr=0.001, 
                 batch_size=512, use_dropout=True,
                 session_key='SessionId', item_key='ItemId', time_key=None, seed=42):
        self.factors = factors
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.epochs_patience = epochs_patience
        self.lr = lr
        self.batch_size=batch_size
        self.use_dropout=use_dropout
        self.seed=seed
        
        self.session_key = session_key
        self.item_key = item_key
        
        self.session = -1
        self.session_items = list()
        
        self.floatX = theano.config.floatX
    
    def fit(self, data, test=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        
        nis = data[self.item_key].nunique()
        
        self.itemmap = pd.Series( index=data[self.item_key].unique(), data=range(1,nis+1)  )
        data = data.merge( self.itemmap.to_frame('ItemIdx'), how='inner', right_index=True, left_on=self.item_key )
        data.sort_values( ['SessionId', 'Time'], inplace=True )
        
        self.traindata = self.create_training_data(data)
        self.dataload = (self.load_data, self.prepare_data)
        self.layers = {'gru': (self.param_init_gru, self.gru_layer)}
        
        self.train_gru(self.factors, self.hidden_units, max_epochs=self.epochs, patience=self.epochs_patience, lrate=self.lr, n_items=nis+1,
                       batch_size=self.batch_size, use_dropout=self.use_dropout)
    
    def train_gru(self, 
    dim_proj=50,  # embeding dimension
    hidden_units=100,  # GRU number of hidden units.
    patience=5,  # Number of epoch to wait before early stop if no progress
    max_epochs=30,  # The maximum number of epoch to run
    dispFreq=10000,  # Display to stdout the training progress every N updates
    lrate=0.001,  # Learning rate
    n_items=37484,  # Vocabulary size
    encoder='gru',  # TODO: can be removed must be gru.
    saveto='gru_model.npz',  # The best model will be saved there
    is_valid=True,  # Compute the validation error after this number of update.
    is_save=False,  # Save the parameters after every saveFreq updates
    batch_size=512,  # The batch size during training.
    valid_batch_size=512,  # The batch size used for validation/test set.
    # Parameter for extra option
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    ):
    
        # Model options
        model_options = locals().copy()
        print("model options", model_options)
    
        load_data, prepare_data = self.get_dataset()
    
        print('Loading data')
        train, valid = load_data()
    
        print('Building model')
        # This create the initial parameters as numpy ndarrays.
        # Dict name (string) -> numpy ndarray
        params = self.init_params(model_options)
    
        if reload_model:
            self.load_params('gru_model.npz', params)
    
        # This create Theano Shared Variable from the parameters.
        # Dict name (string) -> Theano Tensor Shared Variable
        # params and tparams have different copy of the weights.
        tparams = self.init_tparams(params)
    
        # use_noise is for dropout
        (use_noise, x, mask,
         y, f_pred_prob, cost) = self.build_model(tparams, model_options)
         
        self.pred_function = f_pred_prob
         
        all_params = list(tparams.values())
    
        updates = self.adam(cost, all_params, lrate)
    
        train_function = theano.function(inputs=[x, mask, y], outputs=cost, updates=updates)
    
        print('Optimization')
    
        print("%d train examples" % len(train[0]))
        print("%d valid examples" % len(valid[0]))
    
        history_errs = []
        history_vali = []
        best_p = None
        bad_count = 0
    
        uidx = 0  # the number of update done
        estop = False  # early stop
    
        try:
            for eidx in range(max_epochs):
                start_time = time.time()
                n_samples = 0
                epoch_loss = []
    
                # Get new shuffled index for the training set.
                kf = self.get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
                kf_valid = self.get_minibatches_idx(len(valid[0]), valid_batch_size, shuffle=True)
    
                for _, train_index in kf:
                    uidx += 1
                    use_noise.set_value(1.)
    
                    # Select the random examples for this minibatch
                    y = [train[1][t] for t in train_index]
                    x = [train[0][t]for t in train_index]
    
                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, y)
                    n_samples += x.shape[1]
    
                    loss = train_function(x, mask, y)
                    epoch_loss.append(loss)
    
                    if np.isnan(loss) or np.isinf(loss):
                        print('bad loss detected: ', loss)
                        return 1., 1., 1.
    
                    if np.mod(uidx, dispFreq) == 0:
                        print('Epoch ', eidx, 'Update ', uidx, 'Loss ', np.mean(epoch_loss))
    
                if saveto and is_save:
                    print('Saving...')
    
                    if best_p is not None:
                        params = best_p
                    else:
                        params = self.unzip(tparams)
                    np.savez(saveto, history_errs=history_errs, **params)
                    print('Saving done')
    
                if is_valid:
                    use_noise.set_value(0.)
    
                    valid_evaluation = self.pred_evaluation(f_pred_prob, prepare_data, valid, kf_valid)
                    history_errs.append([valid_evaluation])
    
                    if best_p is None or valid_evaluation[1] >= np.array(history_vali).max():
    
                        best_p = self.unzip(tparams)
                        print('Best perfomance updated!')
                        bad_count = 0
    
                    print('Valid Recall@20:', valid_evaluation[0], '   Valid Mrr@20:', valid_evaluation[1])
    
                    if len(history_vali) > 10 and valid_evaluation[1] <= np.array(history_vali).max():
                        bad_count += 1
                        print('===========================>Bad counter: ' + str(bad_count))
                        print('current validation mrr: ' + str(valid_evaluation[1]) +
                              '      history max mrr:' + str(np.array(history_vali).max()))
                        if bad_count > patience:
                            print('Early Stop!')
                            estop = True
    
                    history_vali.append(valid_evaluation[1])
    
                end_time = time.time()
                print('Seen %d samples' % n_samples)
                print(('This epoch took %.1fs' % (end_time - start_time)), file=sys.stderr)
    
                if estop:
                    break
    
        except KeyboardInterrupt:
            print("Training interupted")
    
        if best_p is not None:
            self.zipp(best_p, tparams)
        else:
            best_p = self.unzip(tparams)
    
        use_noise.set_value(0.)
        valid_evaluation = self.pred_evaluation(f_pred_prob, prepare_data, valid, kf_valid)
    
        print('=================Best performance=================')
        print('Valid Recall@20:', valid_evaluation[0], '   Valid Mrr@20:', valid_evaluation[1])
        print('==================================================')
        if saveto and is_save:
            np.savez('Best_performance', valid_evaluation=valid_evaluation, history_errs=history_errs,
                     **best_p)
            
        self.params = params
        self.tparams = tparams
        
        return valid_evaluation
    
    def create_training_data(self, data):
        
        index_session = data.columns.get_loc( self.session_key )
        index_item = data.columns.get_loc( 'ItemIdx' )
        
        out_seqs = []
        labs = []
        
        session = -1
        session_items = []
        
        for row in data.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                session = row[index_session]
                session_items = list()
                
            session_items.append(row[index_item])
            
            if len( session_items ) > 1:
                out_seqs += [ session_items[:-1] ]
                labs += [ session_items[-1] ]
        
        return out_seqs, labs
    
    def prepare_data(self, seqs, labels):
        """Create the matrices from the datasets.
        This pad each sequence to the same lenght: the lenght of the
        longuest sequence or maxlen.
        if maxlen is set, we will cut all sequence to this maximum
        lenght.
        This swap the axis!
        """
        # x: a list of sentences
    
        lengths = [len(s) for s in seqs]
        n_samples = len(seqs)
        maxlen = np.max(lengths)
    
        x = np.zeros((maxlen, n_samples)).astype('int64')
        x_mask = np.ones((maxlen, n_samples)).astype(self.floatX)
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
    
        x_mask *= (1 - (x == 0))
    
        return x, x_mask, labels


    def load_data(self, valid_portion=0.05, maxlen=None, sort_by_len=False):
        '''Loads the dataset
        :type path: String
        :param path: The path to the dataset (here RSC2015)
        :type n_items: int
        :param n_items: The number of items.
        :type valid_portion: float
        :param valid_portion: The proportion of the full train set used for
            the validation set.
        :type maxlen: None or positive int
        :param maxlen: the max sequence length we use in the train/valid set.
        :type sort_by_len: bool
        :name sort_by_len: Sort by the sequence lenght for the train,
            valid and test set. This allow faster execution as it cause
            less padding per minibatch. Another mechanism must be used to
            shuffle the train set at each epoch.
        '''
    
        #############
        # LOAD DATA #
        #############
    
        train_set = self.traindata
    
        if maxlen:
            new_train_set_x = []
            new_train_set_y = []
            for x, y in zip(train_set[0], train_set[1]):
                if len(x) < maxlen:
                    new_train_set_x.append(x)
                    new_train_set_y.append(y)
                else:
                    new_train_set_x.append(x[:maxlen])
                    new_train_set_y.append(y)
            train_set = (new_train_set_x, new_train_set_y)
            del new_train_set_x, new_train_set_y
    
        # split training set into validation set
        train_set_x, train_set_y = train_set
        n_samples = len(train_set_x)
        sidx = np.arange(n_samples, dtype='int32')
        np.random.shuffle(sidx)
        n_train = int(np.round(n_samples * (1. - valid_portion)))
        valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
        valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
        train_set_x = [train_set_x[s] for s in sidx[:n_train]]
        train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    
        train_set = (train_set_x, train_set_y)
        valid_set = (valid_set_x, valid_set_y)
    
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set
    
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    
        if sort_by_len:
            sorted_index = len_argsort(valid_set_x)
            valid_set_x = [valid_set_x[i] for i in sorted_index]
            valid_set_y = [valid_set_y[i] for i in sorted_index]
    
        train = (train_set_x, train_set_y)
        valid = (valid_set_x, valid_set_y)
            
        return train, valid
    
    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        """
        Used to shuffle the dataset at each iteration.
        """
    
        idx_list = np.arange(n, dtype="int32")
    
        if shuffle:
            np.random.shuffle(idx_list)
    
        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                        minibatch_start + minibatch_size])
            minibatch_start += minibatch_size
    
        if minibatch_start != n:
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])
    
        return zip(range(len(minibatches)), minibatches)
    
    def get_dataset(self):
        return self.dataload[0], self.dataload[1]
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, timestamp=0, skip=False, type='view'):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
        if( self.session != session_id ): #new session
            
            self.session = session_id
            self.session_items = list()

        if not input_item_id in self.itemmap.index:
            return None
        
        if type == 'view':
            self.session_items.append( input_item_id )
        
        if skip:
            return
        
        x = [self.itemmap[self.session_items].values]
        y = x
        
        x, mask, y = self.prepare_data(x,y)
        preds = self.pred_function(x, mask)
        
        return pd.Series(data=preds[0][1:], index=self.itemmap.index)


    def zipp(self, params, tparams):
        """
        When we reload the model. Needed for the GPU stuff.
        """
        for kk, vv in params.items():
            tparams[kk].set_value(vv)
    
    
    def unzip(self, zipped):
        """
        When we pickle the model. Needed for the GPU stuff.
        """
        new_params = OrderedDict()
        for kk, vv in zipped.items():
            new_params[kk] = vv.get_value()
        return new_params
    
    
    def dropout_layer(self, state_before, use_noise, trng, drop_p=0.5):
        retain = 1. - drop_p
        proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape,
                                                                 p=retain, n=1,
                                                                 dtype=state_before.dtype)), state_before * retain)
        return proj
    
    
    def _p(self, pp, name):
        return '%s_%s' % (pp, name)
    
    
    def init_params(self, options):
        """
        Global (not GRU) parameter. For the embeding and the classifier.
        """
        params = OrderedDict()
        # embedding
        params['Wemb'] = self.init_weights((options['n_items'], options['dim_proj']))
        params = self.get_layer(options['encoder'])[0](options,
                                                  params,
                                                  prefix=options['encoder'])
        # attention
        params['W_encoder'] = self.init_weights((options['hidden_units'], options['hidden_units']))
        params['W_decoder'] = self.init_weights((options['hidden_units'], options['hidden_units']))
        params['bl_vector'] = self.init_weights((1, options['hidden_units']))
        # classifier
        # params['U'] = init_weights((2*options['hidden_units'], options['n_items']))
        # params['b'] = np.zeros((options['n_items'],)).astype(config.floatX)
        params['bili'] = self.init_weights((options['dim_proj'], 2 * options['hidden_units']))
    
        return params
    
    
    def load_params(self, path, params):
        pp = np.load(path)
        for kk, vv in params.items():
            if kk not in pp:
                raise Warning('%s is not in the archive' % kk)
            params[kk] = pp[kk]
    
        return params
    
    
    def init_tparams(self, params):
        tparams = OrderedDict()
        for kk, pp in params.items():
            tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams
    
    
    def get_layer(self, name):
        fns = self.layers[name]
        return fns
    
    
    def init_weights(self, shape):
        sigma = np.sqrt(2. / shape[0])
        return self.numpy_floatX(np.random.randn(*shape) * sigma)
    
    
    def ortho_weight(self, ndim):
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype(self.floatX)
    
    
    def param_init_gru(self, options, params, prefix='gru'):
        """
        Init the GRU parameter:
    
        :see: init_params
        """
        Wxrz = np.concatenate([self.init_weights((options['dim_proj'], options['hidden_units'])),
                               self.init_weights((options['dim_proj'], options['hidden_units'])),
                               self.init_weights((options['dim_proj'], options['hidden_units']))], axis=1)
        params[self._p(prefix, 'Wxrz')] = Wxrz
    
        Urz = np.concatenate([self.ortho_weight(options['hidden_units']),
                              self.ortho_weight(options['hidden_units'])], axis=1)
        params[self._p(prefix, 'Urz')] = Urz
    
        Uh = self.ortho_weight(options['hidden_units'])
        params[self._p(prefix, 'Uh')] = Uh
    
        b = np.zeros((3 * options['hidden_units'],))
        params[self._p(prefix, 'b')] = b.astype(self.floatX)
        return params
    
    
    def gru_layer(self, tparams, state_below, options, prefix='gru', mask=None):
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
    
        assert mask is not None
    
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]
    
        def _step(m_, x_, h_):
            preact = T.dot(h_, tparams[self._p(prefix, 'Urz')])
            preact += x_[:, 0:2 * options['hidden_units']]
    
            z = T.nnet.hard_sigmoid(_slice(preact, 0, options['hidden_units']))
            r = T.nnet.hard_sigmoid(_slice(preact, 1, options['hidden_units']))
            h = T.tanh(T.dot((h_ * r), tparams[self._p(prefix, 'Uh')]) + _slice(x_, 2, options['hidden_units']))
    
            h = (1.0 - z) * h_ + z * h
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
    
            return h
    
        state_below = (T.dot(state_below, tparams[self._p(prefix, 'Wxrz')]) +
                       tparams[self._p(prefix, 'b')])
    
        hidden_units = options['hidden_units']
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=T.alloc(self.numpy_floatX(0.), n_samples, hidden_units),
                                    name=self._p(prefix, '_layers'),
                                    n_steps=nsteps)
        return rval
    
    def adam(self, loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
        """
        ADAM update rules
        Default values are taken from [Kingma2014]
    
        References:
        [Kingma2014] Kingma, Diederik, and Jimmy Ba.
        "Adam: A Method for Stochastic Optimization."
        arXiv preprint arXiv:1412.6980 (2014).
        http://arxiv.org/pdf/1412.6980v4.pdf
        """
    
        updates = OrderedDict()
        all_grads = theano.grad(loss, all_params)
        alpha = learning_rate
        t = theano.shared(np.float32(1).astype(self.floatX))
        b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)
    
        for theta_previous, g in zip(all_params, all_grads):
            m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=self.floatX))
            v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=self.floatX))
    
            m = b1_t*m_previous + (1 - b1_t)*g  # (Update biased first moment estimate)
            v = b2*v_previous + (1 - b2)*g**2   # (Update biased second raw moment estimate)
            m_hat = m / (1-b1**t)               # (Compute bias-corrected first moment estimate)
            v_hat = v / (1-b2**t)               # (Compute bias-corrected second raw moment estimate)
            theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
    
            updates[m_previous] = m
            updates[v_previous] = v
            updates[theta_previous] = theta
        updates[t] = t + 1.
    
        return updates
    
    
    def build_model(self, tparams, options):
        np.random.seed(self.seed)
        trng = RandomStreams(self.seed)
    
        # Used for dropout.
        use_noise = theano.shared(self.numpy_floatX(0.))
    
        x = T.matrix('x', dtype='int64')
        mask = T.matrix('mask', dtype=self.floatX)
        y = T.vector('y', dtype='int64')
    
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]
    
        emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['dim_proj']])
        if options['use_dropout']:
            emb = self.dropout_layer(emb, use_noise, trng, drop_p=0.25)
    
        proj = self.get_layer(options['encoder'])[1](tparams, emb, options,
                                                prefix=options['encoder'],
                                                mask=mask)
    
        def compute_alpha(state1, state2):
            tmp = T.nnet.hard_sigmoid(T.dot(tparams['W_encoder'], state1.T) + T.dot(tparams['W_decoder'], state2.T))
            alpha = T.dot(tparams['bl_vector'], tmp)
            res = T.sum(alpha, axis=0)
            return res
    
        last_h = proj[-1]
    
        sim_matrix, _ = theano.scan(
            fn=compute_alpha,
            sequences=proj,
            non_sequences=proj[-1]
        )
        att = T.nnet.softmax(sim_matrix.T * mask.T) * mask.T
        p = att.sum(axis=1)[:, None]
        weight = att / p
        atttention_proj = (proj * weight.T[:, :, None]).sum(axis=0)
    
        proj = T.concatenate([atttention_proj, last_h], axis=1)
    
        if options['use_dropout']:
            proj = self.dropout_layer(proj, use_noise, trng, drop_p=0.5)
    
        ytem = T.dot(tparams['Wemb'], tparams['bili'])
        pred = T.nnet.softmax(T.dot(proj, ytem.T))
        # pred = T.nnet.softmax(T.dot(proj, tparams['U']) + tparams['b'])
    
        f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
        # f_weight = theano.function([x, mask], weight, name='f_weight')
    
        off = 1e-8
        if pred.dtype == 'float16':
            off = 1e-6
    
        cost = -T.log(pred[T.arange(n_samples), y] + off).mean()
    
        return use_noise, x, mask, y, f_pred_prob, cost
    
    def pred_evaluation(self, f_pred_prob, prepare_data, data, iterator):
        """
        Compute recall@20 and mrr@20
        f_pred_prob: Theano fct computing the prediction
        prepare_data: usual prepare_data for that dataset.
        """
        recall = 0.0
        mrr = 0.0
        evalutation_point_count = 0
        # pred_res = []
        # att = []
    
        for _, valid_index in iterator:
            x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                      np.array(data[1])[valid_index])
            preds = f_pred_prob(x, mask)
            # weights = f_weight(x, mask)
            targets = y
            ranks = (preds.T > np.diag(preds.T[targets])).sum(axis=0) + 1
            rank_ok = (ranks <= 20)
            # pred_res += list(rank_ok)
            recall += rank_ok.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            evalutation_point_count += len(ranks)
            # att.append(weights)
    
        recall = self.numpy_floatX(recall) / evalutation_point_count
        mrr = self.numpy_floatX(mrr) / evalutation_point_count
        eval_score = (recall, mrr)
    
        # ff = open('/storage/lijing/mydataset/res_attention_correct.pkl', 'wb')
        # pickle.dump(pred_res, ff)
        # ff.close()
        # ff2 = open('/storage/lijing/mydataset/attention_weights.pkl', 'wb')
        # pickle.dump(att, ff2)
        # ff2.close()
    
        return eval_score
    
    def numpy_floatX(self, data):
        return np.asarray(data, dtype=self.floatX)
    
    def clear(self):
        if hasattr(self, 'tparams'): 
            for kk, vv in self.tparams.items():
                if len( self.params[kk].shape ) == 1:
                    self.tparams[kk].set_value([])
                else:
                    self.tparams[kk].set_value([[]])
