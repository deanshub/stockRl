import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'

const getActionIndex = (action) => {
  if (action==='BUY') return [0,1]
  if (action==='STAND') return [1,0]
  if (action==='SELL') return [1,1]
  throw new Error(`Action "${action}" not implemented`)
}

export default class Agent {
  constructor({learningRate = 0.01, goalSteps = 500, scoreRequirement = 50, initialGames = 1000} = {}) {
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

    model.add(tf.layers.dense({ units: 2, activation: 'softmax'}))
    // Specify the loss type and optimizer for training.
    model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: tf.train.adam({learningRate: this.learningRate}),
      // optimizer: 'adam',
      metrics: ['accuracy'],
    })
    return model
  }

  async trainModel(trainingData, epochs = 100, verbose = false) {
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


    const xs = xys.xs // scale?
    const ys = xys.ys // scale?
    const model = this.createModel()
    console.log([xs.length/51, 51], ys.length);
    // prevState.history.length (windowSize) + prevState.stocksBalance.length (1)
    // tf.tensor2d(xs, [xs.length/51, 51]).print()
    // tf.tensor2d(ys, [ys.length, 1]).print()
    await model.fit(tf.tensor2d(xs, [xs.length/51, 51]), tf.tensor2d(ys, [ys.length, 2]), { epochs, verbose })
    // await model.fit(tf.tensor2d(Array.from(Array(51*3)).fill(0), [3, 51]), tf.tensor2d([[0,1],[1,0],[1,1]], [3, 2]), { epochs })
    console.log('model done fitting');
    return model
  }

  async predict(model, xs) {
    const prediction = model.predict(xs)
    // argmax()
    const ys = await prediction.data()
    return ys
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
