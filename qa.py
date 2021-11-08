# -*- coding: utf-8 -*-

import collections
import json

import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import grpc

from create_input import process_input
from create_output import process_output


def communication(hostport, predict_file, model_name):
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    model_request = predict_pb2.PredictRequest()
    model_request.model_spec.name = model_name
    string_record = tf.python_io.tf_record_iterator(path=predict_file)
    model_request.inputs['examples'].CopyFrom(
        tf.contrib.util.make_tensor_proto(next(string_record),
                                          dtype=tf.string,
                                          shape=[1])
    )
    result_future = stub.Predict.future(model_request, 30.0)
    raw_result = result_future.result().outputs

    return raw_result


def process_result(result):
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])

    unique_id = int(result["unique_ids"].int64_val[0])
    start_logits = [float(x) for x in result["start_logits"].float_val]
    end_logits = [float(x) for x in result["end_logits"].float_val]

    # start_logits = np.array(start_logits).reshape(batch_size, max_seq_length)
    # end_logits = np.array(end_logits).reshape(batch_size, max_seq_length)

    formatted_result = RawResult(
        unique_id=unique_id,
        start_logits=start_logits,
        end_logits=end_logits)

    return formatted_result


def mrc(context, question):
    eval_examples, eval_features = process_input(context=context,
                                                 question=question,
                                                 vocab="vocab.txt")

    result = communication(hostport="192.168.0.46:8500",
                           predict_file="test.tf_record",
                           model_name="korquad")
    # print("[result]", result)

    formatted_result = process_result(result)
    # print("[formatted_result]", formatted_result)

    response = process_output([formatted_result],
                         eval_examples,
                         eval_features,
                         question)[0]
    # print("[response]", response)

    # json_res = json.dumps(response)
    # print(json_res)
    # response = json.loads(json_res)

    # return json_res
    return response


if __name__ == '__main__':
    context = "1839년 바그너는 괴테의 파우스트을 처음 읽고 그 내용에 마음이 끌려 이를 소재로 해서 하나의 교향곡을 쓰려는 뜻을 갖는다. 이 시기 바그너는 1838년에 빛 독촉으로 산전수전을 다 걲은 상황이라 좌절과 실망에 가득했으며 메피스토펠레스를 만나는 파우스트의 심경에 공감했다고 한다. 또한 파리에서 아브네크의 지휘로 파리 음악원 관현악단이 연주하는 베토벤의 교향곡 9번을 듣고 깊은 감명을 받았는데, 이것이 이듬해 1월에 파우스트의 서곡으로 쓰여진 이 작품에 조금이라도 영향을 끼쳤으리라는 것은 의심할 여지가 없다. 여기의 라단조 조성의 경우에도 그의 전기에 적혀 있는 것처럼 단순한 정신적 피로나 실의가 반영된 것이 아니라 베토벤의 합창교향곡 조성의 영향을 받은 것을 볼 수 있다. 그렇게 교향곡 작곡을 1839년부터 40년에 걸쳐 파리에서 착수했으나 1악장을 쓴 뒤에 중단했다. 또한 작품의 완성과 동시에 그는 이 서곡(1악장)을 파리 음악원의 연주회에서 연주할 파트보까지 준비하였으나, 실제로는 이루어지지는 않았다. 결국 초연은 4년 반이 지난 후에 드레스덴에서 연주되었고 재연도 이루어졌지만, 이후에 그대로 방치되고 말았다. 그 사이에 그는 리엔치와 방황하는 네덜란드인을 완성하고 탄호이저에도 착수하는 등 분주한 시간을 보냈는데, 그런 바쁜 생활이 이 곡을 잊게 한 것이 아닌가 하는 의견도 있다."
    question = "바그너는 괴테의 파우스트를 읽고 무엇을 쓰고자 했는가?"
    result = mrc(context, question)
    print(result)
