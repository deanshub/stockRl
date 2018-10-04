import { lookup, history } from 'yahoo-stocks'
import moment from 'moment'
// import * as linearModel from './linearModel'
import * as qModel from './qModel'
import {butifyNumber, diffPercentage} from './utils'
const MODEL_TYPES = {
  LINEAR: 'LINEAR',
  Q: 'Q',
}

function isValidInterval(interval) {
  return interval === '1m' ||
    interval === '1d' ||
    interval === '5d' ||
    interval === '1mo' ||
    interval === '1y'
}

function isValidRange(range) {
  return range === '1d' ||
    range === '5d' ||
    range === '1mo' ||
    range === '3mo' ||
    range === '6mo' ||
    range === '1y' ||
    range === '2y' ||
    range === '5y' ||
    range === '10y' ||
    range === 'ytd' ||
    range === 'max'
}

function getData(symbol, {interval = '1d', range = '1y'} = {}){
  if (!symbol) {
    return Promise.reject(new Error('Stock must be defined'))
  } else if (interval==='now'){
    return lookup(symbol)
  } else if (isValidInterval(interval) && isValidRange(range)) {
    return history(symbol, {interval, range})
  } else {
    return Promise.reject(new Error('Interval or Range not valid'))
  }
}

async function learnAndPredict(symbol, options, daysToPredict = 3, modelType = MODEL_TYPES.LINEAR) {
  const data = await getData(symbol, options)

  const stock = {
    symbol,
    records: data.records,
  }

  const predictionsDays = Array.from(Array(daysToPredict)).map((_, index) => (
    moment().add(index, 'days').unix()
  ))

  // if (modelType === MODEL_TYPES.LINEAR) {
  //   const model = await linearModel.stockTrain(stock)
  //   return linearModel.predict(model, predictionsDays)
  // } else
  if (modelType === MODEL_TYPES.Q) {
    const model = await qModel.stockTrain(stock)
    return qModel.predict(model, stock)
  } else {
    return Promise.reject(new Error(`Model type "${modelType}" is not recognized`))
  }
}

function mapMargins(currentPrice, predictions) {
  return predictions.map(prediction => diffPercentage(currentPrice, prediction))
}

function hasUnderMargin(margins, percentMargin = 5) {
  const foundWarning = margins.find(margin => margin < percentMargin)
  return foundWarning
}

const dataOptions = {
  interval: '1d', //1m, 1d, 5d, 1mo, 1y
  range: '10y', // 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
}
const symbol = 'FB'

async function init() {
  const predictions = await learnAndPredict(symbol, dataOptions, 3, MODEL_TYPES.Q)
  predictions.forEach((act,index)=>{
    if (act!=='STAND'){
      console.log(act, index)
    }
  })
  console.log(predictions.length);
  // console.log(predictions);
  // const data = await getData(symbol, {interval: 'now'})
  // const margins = mapMargins(data.currentPrice, predictions)
  // const underMargin = hasUnderMargin(margins)
  // console.log({predictions}, underMargin, margins.map(margin => `${butifyNumber(margin)}%`))
}

init().catch(console.error)
