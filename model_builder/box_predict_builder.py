#-*-coding:utf-8-*-


from box_predictor import head
from box_predictor import box_predictor

def box_predictor_build(istrain,num_class,use_dropout,kernel_size,box_coder_size,dropout_keep_prob,
                          conv_hyperparams_fn,apply_sigmoid_to_scores=False,
                          class_prediction_bias_init=0.0,confidence_prediction_bias_init=0.0,use_depthwise=False):
    box_prediction_head = head.ConvolutionalBoxHead(
        istrain=istrain,
        box_code_size=box_coder_size,
        kernel_size=kernel_size,
        use_depthwise=use_depthwise)
    class_prediction_head = head.ConvolutionalClassHead(
        istrain=istrain,
        num_class=num_class,
        use_dropout=use_dropout,
        dropout_keep_prob=dropout_keep_prob,
        kernel_size=kernel_size,
        apply_sigmoid_to_scores=apply_sigmoid_to_scores,
        bias=class_prediction_bias_init,
        use_depthwise=use_depthwise)
    confidence_prediction_head = head.ConvolutionalConfidenceHead(
        istrain = istrain,
        kernel_size = kernel_size,
        bias = confidence_prediction_bias_init,
        apply_sigmoid_to_scores=apply_sigmoid_to_scores,
        use_depthwise=use_depthwise
    )
    return box_predictor.Box_prediction(istrain=istrain,hyperparamsfn=conv_hyperparams_fn,
                                        class_head=class_prediction_head,box_head=box_prediction_head,confidence_head=confidence_prediction_head)