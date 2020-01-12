import tensorflow as tf



def mask_coverage_loss(attn_dists,coverages, padding_mask):
    '''
    attn_dists: shape (max_len_y, batch_size, max_len_x,1)
    coverages: shape  as above
    padding_mask: shape (batch_size, max_len_y)
    '''
    cov_losses = []
    attn_dists = tf.squeeze(attn_dists, axis = 3)
    coverages = tf.squeeze(coverages, axis = 3)
    for t in range(attn_dists.shape[1]):

        cov_loss_ = tf.reduce_sum(tf.minimum(attn_dists[t,:,:], coverages[t,:,:]),axis =-1)
        cov_losses.append(cov_loss_)
    #(max_len_y, batch_size)->(batch_size, max_len_y)
    cov_losses = tf.stack(cov_losses, axis=1)
    #mask = tf.cast(padding_mask, dtype = loss_.dtype)
    mask = tf.cast(padding_mask, dtype=cov_losses.dtype)
    cov_losses *= mask
    loss = tf.reduce_sum(tf.reduce_mean(cov_losses, axis=0))  # mean loss of each time step and then sum up
    tf.print('coverage loss(batch sum):', loss)
    return loss