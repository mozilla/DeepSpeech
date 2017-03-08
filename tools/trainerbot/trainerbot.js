#!/usr/bin/env node
const fs = require('fs')
const irc = require('irc')
const request = require('request')
const process = require('process')
const program = require('commander')

program
    .version('0.0.1')
    .arguments('<host> <port> <channel> <logfile>')
    .option('-sn, --short <name>', 'short name of the bot', 'trainerbot')
    .option('-fn, --full <name>', 'full name of the bot', 'IRC Trainer Bot')
    .parse(process.argv)

if (program.args.length < 4) {
    program.help()
}

const HOST = program.args[0]
const PORT = program.args[1]
const CHANNEL = program.args[2]
const LOGFILE = program.args[3]
const NAME = program.short
const FULL_NAME = program.full

let epochs = [],
    oldepochs = [],
    users = {},
    juststarted = true

function createGist(description, filename, body, callback) {
    var message = { public: true }
    var files = message.files = { }
    files[filename] = { content: body }
    message.description = description
    request.post({
        uri: 'https://api.github.com/gists',
        headers: { 'User-Agent': [NAME, HOST, PORT, CHANNEL].join(' ') },
        body: JSON.stringify(message)
    }, (err, res, body) => callback && (err ? callback(err) : callback(err, JSON.parse(body))))
}

function createReport(body, callback) {
    return createGist(NAME + ' report', 'report.txt', body, (err, res) => {
        if (err)
            console.error('Problem: ' + err)
        else
            console.log(res)
        callback && callback(err, res)
    })
}

let bot = new irc.Client(HOST, NAME, {
    userName: NAME,
    realName: FULL_NAME,
    port: PORT,
    debug: true,
    secure: false,
    channels: [CHANNEL]
})

function sendReport(caption, body, to) {
    createReport(
        body,
        (err, res) => err ?
            console.error(err) :
            bot.say(to || CHANNEL, (caption ? (caption + ' ') : '') + res.html_url + '\n')
    )
}

function sendEpochReport(caption, epoch, to) {
    sendReport(caption, epoch.lines.join('\n'), to)
}

function getDuration(datediff) {
    let c = Math.abs(datediff)
    let ms = c % 1000
    c = Math.floor(c / 1000)
    let s = c % 60
    c = Math.floor(c / 60)
    let m = c % 60
    c = Math.floor(c / 60)
    let h = c % 24
    let d = Math.floor(c / 24)
    return {
        positive: datediff > 0,
        ms: ms, s: s, m: m, h: h, d: d ,
        dhm: d + ' days ' + h + ' hours ' + m + ' minutes'
    }
}

function percent(wer) {
    return (wer * 100.0).toFixed(2) + '%'
}

bot.addListener('error', message => {
    console.error('ERROR: %s: %s', message.command, message.args.join(' '))
})

bot.addListener('message', (from, to, message) => {
    console.log('%s => %s: %s', from, to, message)
    let lcm = message.toLowerCase()
    if (to.match(/^[#&]/)) {
        // channel message
        if (lcm.match(/hello/)) {
            bot.say(to, 'Hello there ' + from)
        } else if (message.includes(NAME + ':')) {
            if (lcm.match(/help/)) {
                sendReport(
                    null,
                    'last      show the last (finished) epoch\n' +
                    'eta       estimate remaining time of the current epoch\n' +
                    'epoch     show the current epoch\n' +
                    'epoch n   show the n-th epoch\n' +
                    'stats     print a status report with statistics on current training\n' +
                    'csv       print a CSV report of all epoch results of current training\n' +
                    'add       add me to epoch notifications\n' +
                    'remove    remove me from epoch notifications',
                    to
                )
            } else {
                let res = /(last|eta|epoch)/.exec(lcm)
                if (res) {
                    let cmd = res[1]
                    if (epochs.length > 0) {
                        let current = epochs[epochs.length - 1]
                        if (cmd == 'epoch') {
                            res = /epoch (\d+)/.exec(lcm)
                            if (res) {
                                let index = res[1] * 1,
                                    epoch = epochs.find(e => e.index == index)
                                if (epoch) {
                                    sendEpochReport(null, epoch, to)
                                } else {
                                    bot.say(to, 'Sorry, but there is no epoch with that index.')
                                }
                            } else {
                                sendEpochReport(null, current, to)
                            }
                        } else if (cmd == 'last') {
                            if (current.finished) {
                                sendEpochReport(null, current, to)
                            } else if(epochs.length < 2) {
                                bot.say(to, 'Sorry, but there is no finished epoch yet.')
                            } else {
                                sendEpochReport(null, epochs[epochs.length - 2], to)
                            }
                        } else if (cmd == 'eta') {
                            if (current.finished) {
                                bot.say(to, 'There is no running epoch.')
                            } else if(epochs.length < 2) {
                                bot.say(to, 'Sorry, but there is no previous epoch to calculate ETA.')
                            } else {
                                let last = epochs[epochs.length - 2]
                                let duration = getDuration(
                                    (last.finished.getTime() - last.started.getTime()) -
                                    (new Date().getTime() - current.started.getTime())
                                )
                                if (duration.positive) {
                                    bot.say(to, 'Current epoch (' + current.index + ') will be finished in about ' + duration.dhm)
                                } else {
                                    bot.say(to, 'Current epoch (' + current.index + ') should be finished since ' + duration.dhm)
                                }
                            }
                        }
                    } else {
                        bot.say(to, 'Sorry, but there are currently no epochs.')
                    }
                } else if (/stats/.test(lcm)) {
                    if (epochs.length < 2) {
                        bot.say(to, 'Too few epochs for any meaningful statistics.')
                    } else {
                        let first = epochs[0]
                        let count = epochs.length
                        let last = epochs[count - 1]
                        if (!last.finished) {
                            count--
                            last = epochs[count - 1]
                        }

                        let str = 'Training statistics for ' + count + ' epochs\n'
                        str += ' - First epoch (' + first.index + ') loss: ' + (first.training ? first.training.loss.toFixed(2) : '?') + '\n'
                        str += ' - Last epoch (' + last.index + ') loss: ' + (last.training ? last.training.loss.toFixed(2) : '?') + '\n'

                        if(count > 2) {
                            let [a, b, c] = epochs.slice(count - 3, 3).map(e => e.training.loss),
                                d1 = b - a,
                                d2 = c - b
                            if (d1 != 0) {
                                let factor = d2 / d1
                                str += ' - Training loss decrease factor (from epoch ' + a.index + ' to ' + c.index + '): ' + factor.toFixed(2) + '\n'
                                str += ' - Estimated training loss for epoch ' + (c.index + 1) + ': ' + (factor  * c).toFixed(2) + '\n'
                            }
                        }

                        let repochs = epochs.slice(0).reverse(),
                            epoch
                        if (epoch = repochs.find(e => e.validation && 'wer' in e.validation)) {
                            str += ' - Last Validation WER (epoch ' + epoch.index + '): ' + percent(epoch.validation.wer) + '\n'
                        }
                        if (epoch = repochs.find(e => e.training && 'wer' in e.training)) {
                            str += ' - Last Training WER (epoch ' + epoch.index + '): ' + percent(epoch.training.wer) + '\n'
                        }

                        str += ' - Overall time: ' + getDuration(last.finished - first.started).dhm + '\n'
                        str += ' - Mean epoch time: ' + getDuration((last.finished - first.started) / count).dhm

                        sendReport(null, str, to)
                    }
                } else if (/csv/.test(lcm)) {
                    let str = 'epoch,train_loss,train_acc,train_wer,dev_loss,devacc,dev_wer\n'
                    let empty = { loss: '', accuracy: '', wer: '' }
                    for (ei in epochs) {
                        let epoch = epochs[ei],
                            training = epoch.training || empty,
                            validation = epoch.validation || empty
                        str += [
                            epoch.index,
                            training.loss, training.accuracy, training.wer,
                            validation.loss, validation.accuracy, validation.wer
                        ].join(',') + '\n'
                    }
                    sendReport(null, str, to)
                } else if (/add/.test(lcm)) {
                    users[from] = true
                    bot.say(to, from + ': You got added to the notification list.')
                } else if (/remove/.test(lcm)) {
                    if(users[from]) {
                        delete users[from]
                        bot.say(to, from + ': You got removed from the notification list.')
                    }
                }
            }
        }
    }
    else {
        // private message
        console.log('private message')
    }
})

/*
bot.addListener('pm', (nick, message) => {
    console.log('Got private message from %s: %s', nick, message)
})

bot.addListener('join', (channel, who) => {
    console.log('%s has joined %s', who, channel)
})
*/

const sepox = /STARTING Epoch 0*(\d*)/
const fepox = /FINISHED Epoch 0*(\d*)/
const datex = /(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})/
const resex = /([a-zA-Z]+) loss=([0-9\.]+)/
const werex = /avg_cer=([0-9\.]+) WER: ([0-9\.]+)/

function readLogfile() {
    oldepochs = epochs
    epochs = []
    let epoch,
        lines = fs.readFileSync(LOGFILE).toString().split('\n')
    for (i in lines) {
        let header = lines[i].substring(0, 19),
            line = lines[i].substring(20),
            hasdate = datex.exec(header)
        if (hasdate) {
            let [, year, month, day, hours, minutes, seconds] = hasdate,
                date = new Date(year, month-1, day, hours, minutes, seconds)
            if (epoch) {
                if (fepox.test(line)) {
                    epoch.lines.push(line + '      time: ' + date.toString())
                    epoch.finished = date
                    epoch = undefined
                } else {
                    epoch.lines.push(line)
                    let res = resex.exec(line)
                    if (res) {
                        let [, kind, loss, accuracy, wer] = res
                        let v = epoch[kind.toLowerCase()] = { loss: loss * 1.0 }
                        if (res = werex.exec(line)) {
                            v.accuracy = res[1] * 1
                            v.wer = res[2] * 1
                        }
                    }
                }
            } else {
                let res = sepox.exec(line)
                if (res) {
                    epochs.push(epoch = {
                        index: res[1] * 1,
                        started: date,
                        lines: [ line + '      time: ' + date.toString() ]
                    })
                }
            }
        }
    }

    if (epochs.length > 0) {
        if (juststarted) {
            juststarted = false
        } else {
            let lastepoch = epochs[epochs.length - 1]
            if (oldepochs.length > epochs.length) {
                sendEpochReport('Started new training. ' + Object.keys(users).join(', '), lastepoch)
            } else if (
                lastepoch.finished && (
                    // we got another epoch and it's already finished
                    oldepochs.length < epochs.length  ||
                    // the running epoch just finished
                    (oldepochs.length == epochs.length && !oldepochs[oldepochs.length - 1].finished)
                )
            ) {
                sendEpochReport('Just finished one epoch. ' + Object.keys(users).join(', '), lastepoch)
            }
        }
    }
}

fs.watchFile(LOGFILE, (curr, prev) => readLogfile())
readLogfile()
