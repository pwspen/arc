except Exception as e:
        print(f'IDX: {t}, true size: {test_output_shapes}, train output shapes: {train_output_shapes}')
        incorrect.append(t)
        # print(ratio)