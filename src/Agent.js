import path from 'path'
import fs from 'fs-extra'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'

const getActionIndex = (action) => {
  if (action==='BUY') return 0
  if (action==='STAND') return 1
  if (action==='SELL') return 2
  throw new Error(`Action "${action}" not implemented`)
}

export default class Agent {
  constructor({learningRate = 0.0001, goalSteps = 500, scoreRequirement = 50, initialGames = 1000} = {}) {
    this.learningRate = learningRate
    this.goalSteps = goalSteps
    this.scoreRequirement = scoreRequirement
    this.initialGames = initialGames
  }

  generateRandomGames(env, render = false) {
    for (let gameIndex = 0; gameIndex < this.initialGames; gameIndex++) {
      env.reset()
      for (let stepIndex = 0; stepIndex < this.goalSteps; stepIndex++) {
        if (render) {
          env.render()
        }
        const action = env.randomAction()
        const {state, reward, done, info} = env.step(action)
        if (done) {
          return state
        }
      }
    }
  }

  createModel() {
    // A sequential model is a container which you can add layers to.
    const model = tf.sequential()
    // Add a dense layer with 1 output unit.
    model.add(tf.layers.dense({ units: 128, inputShape: [51], activation: 'relu' }))
    model.add(tf.layers.dropout({rate: 0.8}))

    model.add(tf.layers.dense({ units: 256, activation: 'relu' }))
    model.add(tf.layers.dropout({rate: 0.8}))

    model.add(tf.layers.dense({ units: 512, activation: 'relu' }))
    model.add(tf.layers.dropout({rate: 0.8}))

    model.add(tf.layers.dense({ units: 256, activation: 'relu' }))
    model.add(tf.layers.dropout({rate: 0.8}))

    model.add(tf.layers.dense({ units: 128, activation: 'relu' }))
    model.add(tf.layers.dropout({rate: 0.8}))

    model.add(tf.layers.dense({ units: 3, activation: 'softmax'}))
    // Specify the loss type and optimizer for training.
    model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: tf.train.adam(this.learningRate),
      // optimizer: 'adam',
      metrics: ['accuracy'],
    })
    console.log('model created')
    return model
  }

  async trainModel(trainingData, epochs = 1, verbose = false) {
    const modelsDir = path.join(__dirname,'../models')
    try{
      const stat = await fs.stat(path.join(modelsDir,'model.json'))
      if (stat.isFile()){
        console.log('model loaded')
        return tf.loadModel(`file://${path.join(modelsDir,'model.json')}`)
      }
    }catch(e){
      console.log(e)
    }
    const xys = trainingData.reduce((resAll, cur) => {
      const xysOfoneBatch = cur.reduce((res, singleStep) => {
        const {prevState, action, reward, state, done} = singleStep
        res.xs = res.xs.concat(prevState.history.concat([prevState.stocksBalance.length])) //reward?
        // console.log(getActionIndex(action));
        res.ys.push(getActionIndex(action))
        return res
      }, {xs:[], ys:[]})

      resAll.xs = resAll.xs.concat(xysOfoneBatch.xs)
      resAll.ys = resAll.ys.concat(xysOfoneBatch.ys)
      return resAll
    }, {xs:[], ys:[]})


    console.log([xys.xs.length/51, 51], xys.ys.length);
    const xs = tf.tensor2d(xys.xs, [xys.xs.length/51, 51]) // scale?
    // const ys = tf.tensor2d(xys.ys, [xys.ys.length, 3]) // scale?
    const ys = tf.oneHot(tf.tensor1d(xys.ys, 'int32'), 3) // scale?
    const model = this.createModel()
    // prevState.history.length (windowSize) + prevState.stocksBalance.length (1)
    // tf.tensor2d(xs, [xs.length/51, 51]).print()
    // tf.tensor2d(ys, [ys.length, 1]).print()
    await model.fit(xs, ys, { epochs, verbose: false, callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Plot the loss and accuracy values at the end of every training epoch.
        if (verbose){
          console.log(`${epoch+1}) ${logs.loss}`)
        }
      },
    } })
    // await model.fit(tf.tensor2d(Array.from(Array(51*3)).fill(0), [3, 51]), tf.tensor2d([[0,1],[1,0],[1,1]], [3, 2]), { epochs })

    xs.dispose()
    ys.dispose()
    await model.save(`file://${modelsDir}`)
    console.log('model done fitting')
    return model
  }

  async predict(model, x) {
    // console.log(x.length);
    const xs = tf.tensor2d(x, [x.length/51, 51]) // scale?
    // xs.print()
    const prediction = model.predict(xs)
    xs.dispose()
    const ys = await prediction.argMax(-1).data()
    prediction.dispose()
    return Array.from(ys)
  }

  initialPopulation(env, render = false) {
    const trainingData = []
    const scores = []
    const acceptedScores = []

    for (let gameIndex = 0; gameIndex < this.initialGames; gameIndex++) {
      let score = 0
      const gameMemory = []
      let prevState

      env.reset()
      for (let stepIndex = 0; stepIndex < this.goalSteps; stepIndex++) {
        if (render) {
          env.render()
        }
        const action = env.randomAction()
        const {state, reward, done, info} = env.step(action)

        if (prevState) {
          gameMemory.push({prevState, action, reward, state, done})
        }
        prevState = state
        score += reward

        if (done) {
          break
        }
      }

      if (score >= this.scoreRequirement) {
        // console.log(gameMemory[0].prevState.balance, gameMemory[0].action);
        acceptedScores.push(score)
        trainingData.push(gameMemory)
      }

      scores.push(score)
    }

    // console.log({scores, acceptedScores});
    return trainingData
  }
}
