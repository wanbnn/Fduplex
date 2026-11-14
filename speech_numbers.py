"""Portuguese text normalization for TTS; display/history retain numeric notation."""
import re
from decimal import Decimal

# Portuguese thousands use a dot; either comma or dot is accepted for decimals.
NUMBER = r'(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)'


def cardinal(value):
    from num2words import num2words
    return num2words(int(value), lang='pt_BR')


def parts(value):
    if ',' in value:
        return value.replace('.', '').split(',', 1)
    if re.fullmatch(r'\d{1,3}(?:\.\d{3})+', value):
        return value.replace('.', ''), ''
    return value.split('.', 1) if '.' in value else (value, '')


def number(value):
    sign = ''
    if value.startswith(('-', '−', '+')):
        sign = 'menos ' if value[0] in '-−' else 'mais '
        value = value[1:]
    whole, fraction = parts(value)
    # Identifiers with leading zeroes are spoken digit by digit.
    integer = ' '.join(cardinal(digit) for digit in whole) if len(whole) > 1 and whole[0] == '0' else cardinal(whole)
    if fraction:
        integer += ' vírgula ' + ' '.join(cardinal(digit) for digit in fraction)
    return sign + integer


def portuguese_numbers(text):
    if not re.search(r'\d', text):
        return text
    # Remove presentation indices before spelling out numbers.
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', text)

    def currency(match):
        whole, fraction = parts(match[1])
        value = Decimal(whole + '.' + (fraction or '0'))
        from num2words import num2words
        return num2words(value, lang='pt_BR', to='currency')

    text = re.sub(r'R\$\s*(' + NUMBER + r')', currency, text)
    # IPv4 addresses are sequences of components, not a decimal or thousands.
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                  lambda match: ' ponto '.join(cardinal(part) for part in match[0].split('.')), text)
    text = re.sub(r'([-−+]?' + NUMBER + r')\s*%', lambda match: number(match[1]) + ' por cento', text)
    units = {'ms': ('milissegundo', 'milissegundos'), 's': ('segundo', 'segundos'),
             'min': ('minuto', 'minutos'), 'h': ('hora', 'horas'),
             'MB': ('megabyte', 'megabytes'), 'GB': ('gigabyte', 'gigabytes'),
             'MHz': ('megahertz', 'megahertz'), 'GHz': ('gigahertz', 'gigahertz')}
    def unit(match):
        value, suffix = match[1], match[2]
        single, plural = units[suffix]
        words = number(value)
        if suffix == 'h':
            words = re.sub(r'\bum$', 'uma', words)
            words = re.sub(r'\bdois$', 'duas', words)
        return words + ' ' + (single if value in ('1', '1,0', '1.0') else plural)
    text = re.sub(r'([-−+]?' + NUMBER + r')\s*(' + '|'.join(units) + r')\b', unit, text)
    # Avoid converting the digits embedded in ordinary words/model names.
    return re.sub(r'(?<![\w.,])[-−+]?' + NUMBER + r'(?!\w)', lambda match: number(match[0]), text)
