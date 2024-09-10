import yagmail

def send_email(filename):
    # Configurações do Gmail
    sender_email = 'paula.csraissa@gmail.com'
    sender_password = 'smsf xixj nkia zrtb'

    # Crie o objeto de e-mail
    subject = 'Server Inesctec'
    contents = 'Olá, seu codigo terminou de executar ' + filename 
    receiver_email = 'paula.csraissa@gmail.com'

    # Envie o e-mail
    yag = yagmail.SMTP(sender_email, sender_password)
    yag.send(receiver_email, subject, contents)

    print('E-mail enviado com sucesso usando Yagmail!')
