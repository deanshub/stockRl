export function scale(n, start1, stop1, start2, stop2) {
  return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2
}

export function butifyNumber(num) {
  return num % 1 === 0 ? num : parseFloat(num.toFixed(2))
}

export function diffPercentage(from, to) {
  return ((to - from) / from * 100)
}
