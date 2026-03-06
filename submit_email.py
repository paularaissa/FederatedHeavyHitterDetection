import yagmail

def send_email(filename):
    # Configurações do Gmail
    sender_email = ''
    sender_password = ''

    # Crie o objeto de e-mail
    subject = ''
    contents = '' + filename 
    receiver_email = ''

    # Envie o e-mail
    yag = yagmail.SMTP(sender_email, sender_password)
    yag.send(receiver_email, subject, contents)

    print('E-mail enviado com sucesso usando Yagmail!')
