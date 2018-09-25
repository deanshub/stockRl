import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node-gpu'
import {scale} from './utils'

function createModel() {
  // A sequential model is a container which you can add layers to.
  const model = tf.sequential()
  // Add a dense layer with 1 output unit.
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))
  // Specify the loss type and optimizer for training.
  model.compile({ loss: 'meanSquaredError', optimizer: 'SGD' })
  return model
}

export async function train(xs, ys, epochs = 500) {
  const linearModel = createModel()
  await linearModel.fit(tf.tensor2d(xs, [xs.length, 1]), tf.tensor2d(ys, [ys.length, 1]), { epochs })
  return linearModel
}

export async function stockTrain(stock, percentMargin = 5, epochs) {
  const data = stock.records.reduce((res, cur) => {
    if (cur.time && cur.close) {
      const x = cur.time
      const y = cur.close
      if (x < res.minX) {
        res.minX = x
      }
      if (x > res.maxX) {
        res.maxX = x
      }
      if (y < res.minY) {
        res.minY = y
      }
      if (y > res.maxY) {
        res.maxY = y
      }
      res.values.push({
        x,
        y,
      })
    }
    return res
  }, {values: [], minX: Infinity , maxX: -Infinity, minY: Infinity, maxY: -Infinity})

  // scale to more than the maxX and maxY so you can predict to bigger values
  const xMargin = (data.maxX - data.minX) * percentMargin / 100
  const yMargin = (data.maxY - data.minY) * percentMargin / 100
  const scaledData = data.values.reduce((res, {x, y}) => {
    res.xs.push(scale(x, data.minX, data.maxX + xMargin, 0, 1))
    res.ys.push(scale(y, data.minY, data.maxY + yMargin, 0, 1))
    return res
  }, {xs: [], ys: []})

  const model = await train(scaledData.xs, scaledData.ys, epochs)
  return {
    model,
    minX: data.minX,
    maxX: data.maxX + xMargin,
    minY: data.minY,
    maxY: data.maxY + yMargin,
  }
}

export async function predict(fullModel, xs) {
  const { model, minX, maxX, minY, maxY } = fullModel
  const scaledXs = xs.map(x => scale(x, minX, maxX, 0, 1))
  const predictions = model.predict(tf.tensor2d(scaledXs, [scaledXs.length, 1]))
  const ys = await predictions.data()
  return Array.from(ys.map(y => scale(y, 0, 1, minY, maxY)))
}
