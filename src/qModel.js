// import * as tf from '@tensorflow/tfjs'
// import '@tensorflow/tfjs-node'
import Environment from './Environment'
import Agent from './Agent'
import {diffPercentage} from './utils'

class StocksEnv extends Environment {
  constructor(stock, windowSize = 50) {
    super()
    this.stock = stock
    this.windowSize = windowSize
    this.reset()
  }

  reset() {
    this.state = {
      stepInex: 0,
      balance: 1,
      stocksBalance: [],
      history: Array.from(Array(this.windowSize)).fill(0),
    }
  }
  render() {
    console.log(this.state)
  }

  randomAction() {
    // console.log(this.state.balance);
    if (this.state.balance > 0) {
      const actionIndex = Math.floor((Math.random() * 2))
      // console.log(this.actionSpace()[actionIndex]);
      return this.actionSpace()[actionIndex]
    } else {
      const actionIndex = Math.floor((Math.random() * 2) + 1)
      // console.log(this.actionSpace()[actionIndex]);
      return this.actionSpace()[actionIndex]
    }
  }
  actionSpace() {
    return ['BUY','STAND','SELL']
  }
  step(action) {
    const newState = {...this.state}
    let reward = 0
    if (action === 'BUY' && this.state.balance > 0) {
      newState.balance = this.state.balance - 1
      newState.stocksBalance = [...this.state.stocksBalance, this.state.stepInex]
    } else if (action === 'SELL' && this.state.stocksBalance.length > 0) {
      const [buyStepIndex, ...restStockBalance] = this.state.stocksBalance
      newState.stocksBalance = restStockBalance
      reward = diffPercentage(this.stock.records[this.state.stepInex].close, this.stock.records[buyStepIndex].close)
      newState.balance = this.state.balance + 1
    }
    // else if (action === 'STAND') {
    // }
    let historyDiff = 0
    if (this.stock.records[this.state.stepInex-1]) {
      historyDiff = diffPercentage(this.stock.records[this.state.stepInex].close, this.stock.records[this.state.stepInex-1].close)
    }
    const [first, ...restHistory] = this.state.history
    newState.history = [...restHistory, historyDiff]

    newState.stepInex = this.state.stepInex + 1
    const done = this.state.stepInex >= this.stock.records.length
    const info = null
    this.state = newState
    return {state: newState, reward, done, info}
  }
}


export async function stockTrain(stock) {
  const agent = new Agent()
  const env = new StocksEnv(stock)
  // agent.generateRandomGames(env, false)
  const trainingData = agent.initialPopulation(env, false)
  const model = await agent.trainModel(trainingData)
  return {model, agent}
}

function getXsFromStock(stock, windowSize = 50) {
  let xs = []
  for (let recordIndex = windowSize-1; recordIndex < stock.records.length; recordIndex++) {
    xs = xs.concat(stock.records.slice(recordIndex - windowSize, recordIndex))
    xs = xs.concat([0])
  }
  return xs
}

export async function predict(fullModel, stock) {
  const {model, agent} = fullModel
  const xs = getXsFromStock(stock) // 50 days history and balance lenght
  const predictions = await agent.predict(model, xs)
  return predictions
}
